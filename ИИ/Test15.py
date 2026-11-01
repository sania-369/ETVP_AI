import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

# Модель и данные как в [2b]
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
    def __init__(self, dim, hidden):
        super().__init__()
        self.embed = nn.Linear(dim, hidden)
        self.etvp = ETVP_ComplexLayer(hidden, hidden, laplace_mode='none',
                                       R_max=2.0, use_tanh_limit=True,
                                       freeze_theta_mu=True)
        self.head = nn.Linear(hidden, 2)
    def forward(self, x):
        h = self.embed(x)
        h_re, h_im = self.etvp(h, torch.zeros_like(h))
        return self.head(h_re)

def make_data(n=3000, dim=20, noise=1.5, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n, dim)
    w = torch.randn(dim)
    y = ((X @ w + noise * torch.randn(n)) > 0).long()
    return X, y

X_all, y_all = make_data()
X_tr, y_tr = X_all[:2000], y_all[:2000]

torch.manual_seed(0)
model = ETVPModel(20, 64)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

print("Эпоха | loss | W_max | b_max | embed_max | head_max")
print("-" * 65)

for ep in range(150):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(X_tr), y_tr)
    loss = loss + 0.01 * model.etvp.get_z_penalty()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step()

    # Проверить параметры
    w_max = max(model.etvp.W_re.abs().max().item(), model.etvp.W_im.abs().max().item())
    b_max = max(model.etvp.b_re.abs().max().item(), model.etvp.b_im.abs().max().item())
    e_max = max(p.abs().max().item() for p in model.embed.parameters())
    h_max = max(p.abs().max().item() for p in model.head.parameters())

    if ep % 10 == 0 or ep > 95:
        print(f"{ep:5d} | {loss.item():.4f} | {w_max:.3f} | {b_max:.3f} | {e_max:.3f} | {h_max:.3f}")

    # Проверить на NaN
    bad = any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters())
    if bad:
        print(f"\nЭпоха {ep}: NaN в параметрах!")
        for name, p in model.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"  {name}: max = {p.abs().max().item()}")
        break

    if torch.isnan(loss).any():
        print(f"\nЭпоха {ep}: NaN в loss!")
        break
