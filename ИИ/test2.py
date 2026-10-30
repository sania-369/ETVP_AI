import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Literal


# ============================================================
# ETVP Complex Layer v3.2 (из предыдущего сообщения)
# ============================================================
class ETVP_ComplexLayer(nn.Module):
    EPS: float = 1e-8

    def __init__(self, dim_in, dim_out, R_max=1.0, eta=0.01, beta=0.1,
                 laplace_mode='auto', use_cardioid=True,
                 padding_mode='reflect', mode='increment', lambda_z=1.0):
        super().__init__()
        if dim_in != dim_out:
            raise ValueError("dim_in must equal dim_out")
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.R_max = R_max
        self.eta = eta
        self.beta = beta
        self.laplace_mode = laplace_mode
        self.use_cardioid = use_cardioid
        self.padding_mode = padding_mode
        self.mode = mode
        self.lambda_z = lambda_z

        bound = (2.0 / dim_in) ** 0.5 / (2.0 ** 0.5)
        self.W_re = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.W_im = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.b_re = nn.Parameter(0.01 * torch.randn(dim_out))
        self.b_im = nn.Parameter(0.01 * torch.randn(dim_out))
        self.theta_mu = nn.Parameter(torch.tensor(-4.0))

    def get_z_penalty(self):
        W_norm = torch.abs(self.W_re + 1j * self.W_im).norm(dim=0)
        return self.lambda_z * F.relu(W_norm - self.R_max).pow(2).mean()

    def laplacian_spatial(self, z):
        if z.dim() != 4:
            return torch.zeros_like(z)
        _, _, H, W = z.shape
        if H < 2 or W < 2:
            return torch.zeros_like(z)
        pad_map = {'reflect': 'reflect', 'zero': 'constant', 'periodic': 'circular'}
        z_padded = F.pad(z, (1, 1, 1, 1), mode=pad_map.get(self.padding_mode, 'reflect'))
        lap = (z_padded[:, :, :-2, 1:-1] + z_padded[:, :, 2:, 1:-1] +
               z_padded[:, :, 1:-1, :-2] + z_padded[:, :, 1:-1, 2:] - 4 * z)
        return lap / (H * W + self.EPS)

    def laplacian(self, z):
        mode = self.laplace_mode
        if mode == 'auto':
            mode = 'spatial' if z.dim() == 4 else 'none'
        return self.laplacian_spatial(z) if mode == 'spatial' else torch.zeros_like(z)

    def complex_mul(self, x_re, x_im):
        out_re = x_re @ self.W_re - x_im @ self.W_im + self.b_re
        out_im = x_re @ self.W_im + x_im @ self.W_re + self.b_im
        return out_re, out_im

    def cardioid_activation(self, re, im):
        angle = torch.atan2(im, re)
        scale = 0.5 * (1.0 + torch.cos(angle))
        return scale * re, scale * im

    def forward(self, x_re, x_im, dt=0.1, return_intermediate=False):
        f_re, f_im = self.complex_mul(x_re, x_im)
        if self.use_cardioid:
            f_re, f_im = self.cardioid_activation(f_re, f_im)
        else:
            f_re, f_im = torch.tanh(f_re), torch.tanh(f_im)

        mu = F.softplus(self.theta_mu)
        lap_re = self.laplacian(f_re)
        lap_im = self.laplacian(f_im)
        f_abs = torch.sqrt(f_re**2 + f_im**2) + self.EPS
        C = torch.tanh(f_abs.mean()).item()

        sqrt_dt = math.sqrt(dt)
        noise_scale = self.eta * torch.tanh(self.beta * f_abs)
        phase = torch.atan2(f_im, f_re)
        noise_re = sqrt_dt * noise_scale * torch.randn_like(f_re) * torch.cos(phase)
        noise_im = sqrt_dt * noise_scale * torch.randn_like(f_im) * torch.sin(phase)

        k1_re = f_re - mu * lap_re
        k1_im = f_im - mu * lap_im
        x_pred_re = x_re + dt * k1_re
        x_pred_im = x_im + dt * k1_im

        f_pred_re, f_pred_im = self.complex_mul(x_pred_re, x_pred_im)
        if self.use_cardioid:
            f_pred_re, f_pred_im = self.cardioid_activation(f_pred_re, f_pred_im)
        else:
            f_pred_re, f_pred_im = torch.tanh(f_pred_re), torch.tanh(f_pred_im)

        lap_pred_re = self.laplacian(f_pred_re)
        lap_pred_im = self.laplacian(f_pred_im)
        k2_re = f_pred_re - mu * lap_pred_re
        k2_im = f_pred_im - mu * lap_pred_im

        z_re_next = x_re + dt * (k1_re + k2_re) / 2 + noise_re
        z_im_next = x_im + dt * (k1_im + k2_im) / 2 + noise_im

        if return_intermediate:
            return z_re_next, z_im_next, {'C': C, 'mu': mu.item()}
        return z_re_next, z_im_next


# ============================================================
# МОДЕЛИ ДЛЯ СРАВНЕНИЯ
# ============================================================
class BaselineMLP(nn.Module):
    """Стандартный FFN."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.net(x)


class ETVPModel(nn.Module):
    """ЕТВП-слой вместо FFN."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.embed = nn.Linear(dim, hidden)
        self.etvp = ETVP_ComplexLayer(hidden, hidden, laplace_mode='none')
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.embed(x)
        h_re, h_im = self.etvp(h, torch.zeros_like(h))
        return self.head(h_re)


# ============================================================
# ДАННЫЕ: шумная классификация
# ============================================================
def make_data(n=2000, dim=20, noise=1.5, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n, dim)
    # Истинная граница — линейная, но с шумом
    w = torch.randn(dim)
    logits = X @ w + noise * torch.randn(n)
    y = (logits > 0).long()
    return X, y


# ============================================================
# ОБУЧЕНИЕ
# ============================================================
def train(model, X, y, epochs=300, lr=1e-2, use_z_penalty=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        if use_z_penalty and hasattr(model, 'etvp'):
            loss = loss + 0.01 * model.etvp.get_z_penalty()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        acc = (pred == y).float().mean().item()
    return acc


# ============================================================
# ЭКСПЕРИМЕНТ
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    # Train / test split
    X_all, y_all = make_data(n=3000, dim=20, noise=1.5)
    n_train = 2000
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]

    print("=" * 60)
    print("ДАННЫЕ: 20 признаков, шум 1.5, 3000 примеров")
    print(f"Train: {n_train}, Test: {len(X_test)}")
    print("=" * 60)

    # Baseline
    print("\n[1] Baseline MLP")
    baseline = BaselineMLP(dim=20, hidden=64)
    losses_b = train(baseline, X_train, y_train, epochs=300)
    acc_b = evaluate(baseline, X_test, y_test)
    print(f"    Train acc: {evaluate(baseline, X_train, y_train):.4f}")
    print(f"    Test acc:  {acc_b:.4f}")
    print(f"    Final loss: {losses_b[-1]:.4f}")

    # ETVP без Z-штрафа
    print("\n[2] ETVP (без Z-штрафа)")
    etvp1 = ETVPModel(dim=20, hidden=64)
    losses_e1 = train(etvp1, X_train, y_train, epochs=300, use_z_penalty=False)
    acc_e1 = evaluate(etvp1, X_test, y_test)
    print(f"    Train acc: {evaluate(etvp1, X_train, y_train):.4f}")
    print(f"    Test acc:  {acc_e1:.4f}")
    print(f"    Final loss: {losses_e1[-1]:.4f}")

    # ETVP с Z-штрафом
    print("\n[3] ETVP (с Z-штрафом)")
    etvp2 = ETVPModel(dim=20, hidden=64)
    losses_e2 = train(etvp2, X_train, y_train, epochs=300, use_z_penalty=True)
    acc_e2 = evaluate(etvp2, X_test, y_test)
    print(f"    Train acc: {evaluate(etvp2, X_train, y_train):.4f}")
    print(f"    Test acc:  {acc_e2:.4f}")
    print(f"    Final loss: {losses_e2[-1]:.4f}")
    print(f"    Z-penalty: {etvp2.etvp.get_z_penalty().item():.4f}")

    # Диагностика C
    print("\n[Диагностика ETVP]")
    etvp2.eval()
    with torch.no_grad():
        h = etvp2.embed(X_test)
        _, _, inter = etvp2.etvp(h, torch.zeros_like(h), return_intermediate=True)
        print(f"    C на test: {inter['C']:.4f}")
        print(f"    mu:       {inter['mu']:.4f}")

    print("\n" + "=" * 60)
    print("ИТОГ:")
    print(f"  Baseline:      {acc_b:.4f}")
    print(f"  ETVP (no Z):   {acc_e1:.4f}")
    print(f"  ETVP (with Z): {acc_e2:.4f}")
    print("=" * 60)
