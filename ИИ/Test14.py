import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("ТЕСТЫ ETVP v3.6")
print("=" * 60)

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
                 freeze_theta_mu=True):
        super().__init__()
        self.embed = nn.Linear(dim, hidden)
        self.etvp = ETVP_ComplexLayer(
            hidden, hidden, laplace_mode='none', R_max=R_max,
            use_tanh_limit=use_tanh_limit, freeze_theta_mu=freeze_theta_mu
        )
        self.head = nn.Linear(hidden, 2)
    def forward(self, x):
        h = self.embed(x)
        h_re, h_im = self.etvp(h, torch.zeros_like(h))
        return self.head(h_re)


# ============================================================
# [1] Диагностика: 50 шагов, батч 64 (где был NaN на шаге 27)
# ============================================================
print("\n[1] Диагностика: 50 шагов, батч 64")
torch.manual_seed(0)
model = ETVPModel(dim=20, hidden=64)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
X = torch.randn(64, 20)
y = torch.randint(0, 2, (64,))

nan_found = False
for step in range(50):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    bad = any(
        p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
        for p in model.parameters()
    )
    if bad:
        print(f"    Шаг {step}: NaN/Inf в градиентах")
        nan_found = True
        break
    opt.step()

if not nan_found:
    print(f"    OK: 50 шагов без NaN, loss = {loss.item():.4f}")


# ============================================================
# [2] Данные
# ============================================================
def make_data(n=3000, dim=20, noise=1.5, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n, dim)
    w = torch.randn(dim)
    y = ((X @ w + noise * torch.randn(n)) > 0).long()
    return X, y

X_all, y_all = make_data()
n_train = 2000
X_tr, y_tr = X_all[:n_train], y_all[:n_train]
X_te, y_te = X_all[n_train:], y_all[n_train:]


def train(model, epochs=300, lr=1e-2, use_z=False, clip=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        if use_z and hasattr(model, 'etvp'):
            loss = loss + 0.01 * model.etvp.get_z_penalty()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if math.isnan(loss.item()):
            print(f"    NaN на эпохе {ep}")
            return
    print(f"    Готово")


def acc(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


# ============================================================
# [2] Обучение
# ============================================================
print("\n[2] Обучение")
print("\n  [2a] Baseline")
torch.manual_seed(0)
base = BaselineMLP(20, 64)
train(base)
print(f"    Train: {acc(base, X_tr, y_tr):.4f}")
print(f"    Test:  {acc(base, X_te, y_te):.4f}")

print("\n  [2b] ETVP v3.6 (sigmoid + tanh + Z, theta_mu frozen)")
torch.manual_seed(0)
etvp = ETVPModel(20, 64, use_tanh_limit=True, freeze_theta_mu=True)
train(etvp, use_z=True, clip=0.5)
print(f"    Train: {acc(etvp, X_tr, y_tr):.4f}")
print(f"    Test:  {acc(etvp, X_te, y_te):.4f}")

print("\n  [2c] ETVP v3.6 без Z-штрафа (для сравнения)")
torch.manual_seed(0)
etvp2 = ETVPModel(20, 64, use_tanh_limit=True, freeze_theta_mu=True)
train(etvp2, use_z=False, clip=0.5)
print(f"    Train: {acc(etvp2, X_tr, y_tr):.4f}")
print(f"    Test:  {acc(etvp2, X_te, y_te):.4f}")


# ============================================================
# [3] Диагностика
# ============================================================
print("\n[3] Диагностика ETVP")
etvp.eval()
with torch.no_grad():
    h = etvp.embed(X_te)
    _, _, inter = etvp.etvp(h, torch.zeros_like(h), return_intermediate=True)
    print(f"    C: {inter['C']:.4f}")
    print(f"    mu: {inter['mu']:.4f}")
    print(f"    restore max: {inter['restore'][0].abs().max().item():.4f}")
    print(f"    |f| max: {inter['f'][0].abs().max().item():.4f}")


# ============================================================
# [4] Стабильность: 1000 шагов без обучения
# ============================================================
print("\n[4] Стабильность: 1000 шагов без обучения")
layer = ETVP_ComplexLayer(64, 64, laplace_mode='none', R_max=2.0)
x_re = torch.randn(8, 64)
x_im = torch.randn(8, 64)
mags = []
exploded = False
for step in range(1000):
    with torch.no_grad():
        x_re, x_im = layer(x_re, x_im, dt=0.1)
    mag = torch.sqrt(x_re**2 + x_im**2).max().item()
    mags.append(mag)
    if math.isnan(mag) or math.isinf(mag):
        print(f"    ВЗРЫВ на шаге {step}: mag = {mag}")
        exploded = True
        break
if not exploded:
    print(f"    OK: max|z| = {max(mags):.4f}, min = {min(mags):.4f}")


print("\n" + "=" * 60)
print("ИТОГ:")
print(f"  Baseline:        {acc(base, X_te, y_te):.4f}")
print(f"  ETVP (Z):        {acc(etvp, X_te, y_te):.4f}")
print(f"  ETVP (без Z):    {acc(etvp2, X_te, y_te):.4f}")
print("=" * 60)
