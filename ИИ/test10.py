import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Literal


# ============================================================
# ETVP Complex Layer v3.4
# ============================================================
class ETVP_ComplexLayer(nn.Module):
    EPS: float = 1e-8

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 R_max: float = 2.0,
                 eta: float = 0.01,
                 beta: float = 0.1,
                 laplace_mode: Literal['auto', 'spatial', 'features', 'none'] = 'auto',
                 use_cardioid: bool = True,
                 padding_mode: Literal['reflect', 'zero', 'periodic'] = 'reflect',
                 mode: Literal['increment', 'generate'] = 'increment',
                 lambda_z: float = 1.0,
                 k_hyper: float = 30.0,
                 gamma_restore: float = 0.1,
                 use_tanh_limit: bool = True,
                 freeze_theta_mu: bool = False):
        super().__init__()

        if dim_in != dim_out:
            raise ValueError(
                f"ETVP_ComplexLayer требует dim_in == dim_out (получено {dim_in} и {dim_out})."
            )

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
        self.k_hyper = k_hyper
        self.gamma_restore = gamma_restore
        self.use_tanh_limit = use_tanh_limit
        self.freeze_theta_mu = freeze_theta_mu

        bound = (2.0 / dim_in) ** 0.5 / (2.0 ** 0.5)
        self.W_re = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.W_im = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.b_re = nn.Parameter(0.01 * torch.randn(dim_out))
        self.b_im = nn.Parameter(0.01 * torch.randn(dim_out))
        self.theta_mu = nn.Parameter(torch.tensor(-4.0))
        if freeze_theta_mu:
            self.theta_mu.requires_grad = False

    def get_z_penalty(self) -> torch.Tensor:
        W_norm = torch.abs(self.W_re + 1j * self.W_im).norm(dim=0)
        penalty = F.relu(W_norm - self.R_max).pow(2).mean()
        return self.lambda_z * penalty

    def hyperbolic_restore(self, re: torch.Tensor, im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag = torch.clamp(torch.sqrt(re**2 + im**2), min=1e-4)
        C = mag / self.R_max
        E = C / (1.0 + self.k_hyper * C) * (1.0 + self.k_hyper)
        force = self.gamma_restore * (E - C)
        force_re = -force * re / mag
        force_im = -force * im / mag
        return force_re, force_im

    def laplacian_spatial(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4:
            return torch.zeros_like(z)
        _, _, H, W = z.shape
        if H < 2 or W < 2:
            return torch.zeros_like(z)
        pad_map = {'reflect': 'reflect', 'zero': 'constant', 'periodic': 'circular'}
        z_padded = F.pad(z, (1, 1, 1, 1), mode=pad_map.get(self.padding_mode, 'reflect'))
        lap = (z_padded[:, :, :-2, 1:-1] +
               z_padded[:, :, 2:, 1:-1] +
               z_padded[:, :, 1:-1, :-2] +
               z_padded[:, :, 1:-1, 2:] -
               4 * z)
        return lap / (H * W + self.EPS)

    def laplacian(self, z: torch.Tensor) -> torch.Tensor:
        mode = self.laplace_mode
        if mode == 'auto':
            mode = 'spatial' if z.dim() == 4 else 'none'
        if mode == 'spatial':
            return self.laplacian_spatial(z)
        return torch.zeros_like(z)

    def complex_mul(self, x_re: torch.Tensor, x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_re.dim() == 2:
            out_re = x_re @ self.W_re - x_im @ self.W_im + self.b_re
            out_im = x_re @ self.W_im + x_im @ self.W_re + self.b_im
        elif x_re.dim() == 4:
            out_re = (torch.einsum('bihw,io->bohw', x_re, self.W_re) -
                      torch.einsum('bihw,io->bohw', x_im, self.W_im) +
                      self.b_re.view(1, -1, 1, 1))
            out_im = (torch.einsum('bihw,io->bohw', x_re, self.W_im) +
                      torch.einsum('bihw,io->bohw', x_im, self.W_re) +
                      self.b_im.view(1, -1, 1, 1))
        else:
            raise ValueError(f"Unsupported tensor dims: {x_re.dim()}.")
        return out_re, out_im

    def cardioid_activation(self, re: torch.Tensor, im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag = torch.sqrt(re**2 + im**2 + 1e-6)
        cos_angle = re / mag
        scale = 0.5 * (1.0 + cos_angle)
        return scale * re, scale * im

    def forward(self,
                x_re: torch.Tensor,
                x_im: torch.Tensor,
                dt: float = 0.1,
                return_intermediate: bool = False):
        restore_re, restore_im = self.hyperbolic_restore(x_re, x_im)

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

        z_re_next = x_re + dt * (k1_re + k2_re) / 2 + dt * restore_re + noise_re
        z_im_next = x_im + dt * (k1_im + k2_im) / 2 + dt * restore_im + noise_im

        if self.use_tanh_limit:
            z_re_next = self.R_max * torch.tanh(z_re_next / self.R_max)
            z_im_next = self.R_max * torch.tanh(z_im_next / self.R_max)

        if return_intermediate:
            return z_re_next, z_im_next, {
                'f': (f_re, f_im),
                'f_pred': (f_pred_re, f_pred_im),
                'restore': (restore_re, restore_im),
                'mu': mu,
                'C': C,
                'dt': dt
            }
        return z_re_next, z_im_next


# ============================================================
# МОДЕЛИ
# ============================================================
class BaselineMLP(nn.Module):
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
    def __init__(self, dim, hidden, use_tanh_limit=True, R_max=2.0,
                 use_cardioid=True, freeze_theta_mu=False):
        super().__init__()
        self.embed = nn.Linear(dim, hidden)
        self.etvp = ETVP_ComplexLayer(
            hidden, hidden,
            laplace_mode='none',
            R_max=R_max,
            use_tanh_limit=use_tanh_limit,
            use_cardioid=use_cardioid,
            freeze_theta_mu=freeze_theta_mu
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.embed(x)
        h_re, h_im = self.etvp(h, torch.zeros_like(h))
        return self.head(h_re)


# ============================================================
# ДАННЫЕ
# ============================================================
def make_data(n=3000, dim=20, noise=1.5, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n, dim)
    w = torch.randn(dim)
    logits = X @ w + noise * torch.randn(n)
    y = (logits > 0).long()
    return X, y


# ============================================================
# ДИАГНОСТИКА ОБУЧЕНИЯ (ловит NaN по шагам)
# ============================================================
def diagnose_training(model, X, y, steps=50, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(steps):
        model.train()
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()

        # Проверить градиенты
        bad_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None:
                g_max = p.grad.abs().max().item()
                if math.isnan(g_max) or math.isinf(g_max):
                    print(f"    Шаг {step}: NaN/Inf в градиенте {name}")
                    bad_grad = True
        if bad_grad:
            return step, "grad"

        opt.step()

        # Проверить параметры
        for name, p in model.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"    Шаг {step}: NaN/Inf в параметре {name}")
                return step, name

        # Проверить forward
        with torch.no_grad():
            out_check = model(X[:8])
            if torch.isnan(out_check).any():
                print(f"    Шаг {step}: NaN в forward")
                return step, "forward"

        if step % 10 == 0:
            theta = model.etvp.theta_mu.item()
            mu = F.softplus(model.etvp.theta_mu).item()
            w_max = model.etvp.W_re.abs().max().item()
            print(f"    Шаг {step}: loss={loss.item():.4f}, "
                  f"theta_mu={theta:.3f}, mu={mu:.4f}, |W|max={w_max:.3f}")

    return steps, "ok"


# ============================================================
# ОБУЧЕНИЕ
# ============================================================
def train(model, X, y, epochs=300, lr=1e-2, use_z_penalty=False,
          clip_grad=None, weight_decay=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()
        losses.append(loss.item())
        if math.isnan(loss.item()):
            print(f"    NaN на эпохе {ep}")
            break
    return losses


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        acc = (pred == y).float().mean().item()
    return acc


# ============================================================
# ТЕСТЫ
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ТЕСТЫ ETVP v3.4 + ДИАГНОСТИКА")
    print("=" * 60)

    # [0] Версия
    print("\n[0] Проверка версии")
    print("    has hyperbolic_restore:", hasattr(ETVP_ComplexLayer, 'hyperbolic_restore'))
    print("    has freeze_theta_mu:", 'freeze_theta_mu' in ETVP_ComplexLayer.__init__.__code__.co_varnames)

    # [1] Диагностика: где NaN
    print("\n[1] Диагностика обучения (50 шагов, батч 64)")
    torch.manual_seed(0)
    model_diag = ETVPModel(dim=20, hidden=64)
    X_diag = torch.randn(64, 20)
    y_diag = torch.randint(0, 2, (64,))
    step_fail, reason = diagnose_training(model_diag, X_diag, y_diag, steps=50)
    print(f"    Итог: шаг {step_fail}, причина: {reason}")

    # [2] Обучение с разными настройками
    print("\n[2] Обучение: сравнение настроек")
    X_all, y_all = make_data(n=3000, dim=20, noise=1.5)
    n_train = 2000
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]

    configs = [
        ("baseline", None),
        ("ETVP freeze_theta + wd=1e-4 + clip=0.5",
         dict(freeze_theta_mu=True, weight_decay=1e-4, clip_grad=0.5)),
        ("ETVP freeze_theta + tanh + wd=1e-4",
         dict(freeze_theta_mu=True, weight_decay=1e-4, clip_grad=None)),
    ]

    results = {}
    for name, cfg in configs:
        print(f"\n  [{name}]")
        torch.manual_seed(0)
        if name == "baseline":
            model = BaselineMLP(dim=20, hidden=64)
            losses = train(model, X_train, y_train, epochs=300)
        else:
            model = ETVPModel(dim=20, hidden=64, freeze_theta_mu=cfg['freeze_theta_mu'])
            losses = train(model, X_train, y_train, epochs=300,
                           use_z_penalty=True,
                           clip_grad=cfg['clip_grad'],
                           weight_decay=cfg['weight_decay'])
        train_acc = evaluate(model, X_train, y_train)
        test_acc = evaluate(model, X_test, y_test)
        results[name] = (train_acc, test_acc, losses[-1] if losses else float('nan'))
        print(f"    Train: {train_acc:.4f}")
        print(f"    Test:  {test_acc:.4f}")
        print(f"    Final loss: {losses[-1] if losses else 'nan'}")

    # [3] Диагностика лучшего ETVP
    print("\n[3] Диагностика ETVP (лучшая конфигурация)")
    best_etvp = None
    for name, cfg in configs[1:]:
        if best_etvp is None:
            torch.manual_seed(0)
            best_etvp = ETVPModel(dim=20, hidden=64, freeze_theta_mu=cfg['freeze_theta_mu'])
            train(best_etvp, X_train, y_train, epochs=300,
                  use_z_penalty=True,
                  clip_grad=cfg['clip_grad'],
                  weight_decay=cfg['weight_decay'])
    best_etvp.eval()
    with torch.no_grad():
        h = best_etvp.embed(X_test)
        _, _, inter = best_etvp.etvp(h, torch.zeros_like(h), return_intermediate=True)
        print(f"    C: {inter['C']:.4f}")
        print(f"    mu: {inter['mu']:.4f}")
        print(f"    theta_mu: {best_etvp.etvp.theta_mu.item():.4f}")
        print(f"    restore max: {inter['restore'][0].abs().max().item():.4f}")

    print("\n" + "=" * 60)
    print("ИТОГ:")
    for name, (tr, te, loss) in results.items():
        print(f"  {name}: Train={tr:.4f}, Test={te:.4f}, Loss={loss}")
    print("=" * 60)
