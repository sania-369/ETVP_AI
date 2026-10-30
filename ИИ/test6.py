import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# ETVP Complex Layer v3.3 (вставь сюда класс из предыдущего сообщения,
# если он ещё не в ячейке)
# ============================================================

# ============================================================
# МОДЕЛИ ДЛЯ СРАВНЕНИЯ
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
    def __init__(self, dim, hidden, use_tanh_limit=True, R_max=2.0):
        super().__init__()
        self.embed = nn.Linear(dim, hidden)
        self.etvp = ETVP_ComplexLayer(
            hidden, hidden,
            laplace_mode='none',
            R_max=R_max,
            use_tanh_limit=use_tanh_limit
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
# ОБУЧЕНИЕ
# ============================================================
def train(model, X, y, epochs=300, lr=1e-2, use_z_penalty=False, clip_grad=None):
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
# ЭКСПЕРИМЕНТ
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    X_all, y_all = make_data(n=3000, dim=20, noise=1.5)
    n_train = 2000
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]

    print("=" * 60)
    print("ОБУЧЕНИЕ: 20 признаков, шум 1.5, 3000 примеров")
    print("=" * 60)

    # Baseline
    print("\n[1] Baseline MLP")
    baseline = BaselineMLP(dim=20, hidden=64)
    losses_b = train(baseline, X_train, y_train, epochs=300)
    print(f"    Train: {evaluate(baseline, X_train, y_train):.4f}")
    print(f"    Test:  {evaluate(baseline, X_test, y_test):.4f}")
    print(f"    Final loss: {losses_b[-1]:.4f}")

    # ETVP без Z, без tanh
    print("\n[2] ETVP (без Z, без tanh-страховки)")
    etvp1 = ETVPModel(dim=20, hidden=64, use_tanh_limit=False)
    losses_e1 = train(etvp1, X_train, y_train, epochs=300, use_z_penalty=False)
    print(f"    Train: {evaluate(etvp1, X_train, y_train):.4f}")
    print(f"    Test:  {evaluate(etvp1, X_test, y_test):.4f}")
    print(f"    Final loss: {losses_e1[-1]:.4f}")

    # ETVP с Z, с tanh
    print("\n[3] ETVP (с Z, с tanh-страховкой)")
    etvp2 = ETVPModel(dim=20, hidden=64, use_tanh_limit=True)
    losses_e2 = train(etvp2, X_train, y_train, epochs=300,
                       use_z_penalty=True, clip_grad=1.0)
    print(f"    Train: {evaluate(etvp2, X_train, y_train):.4f}")
    print(f"    Test:  {evaluate(etvp2, X_test, y_test):.4f}")
    print(f"    Final loss: {losses_e2[-1]:.4f}")
    print(f"    Z-penalty: {etvp2.etvp.get_z_penalty().item():.4f}")

    # Диагностика
    print("\n[Диагностика ETVP с Z]")
    etvp2.eval()
    with torch.no_grad():
        h = etvp2.embed(X_test)
        _, _, inter = etvp2.etvp(h, torch.zeros_like(h), return_intermediate=True)
        print(f"    C: {inter['C']:.4f}")
        print(f"    mu: {inter['mu']}")
        print(f"    restore max: {inter['restore'][0].abs().max().item():.4f}")

    print("\n" + "=" * 60)
    print("ИТОГ:")
    print(f"  Baseline:        {evaluate(baseline, X_test, y_test):.4f}")
    print(f"  ETVP (no tanh):  {evaluate(etvp1, X_test, y_test):.4f}")
    print(f"  ETVP (with tanh):{evaluate(etvp2, X_test, y_test):.4f}")
    print("=" * 60)
