import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# (класс ETVP_ComplexLayer v3.3 уже в памяти)

torch.manual_seed(0)

model = ETVPModel(dim=20, hidden=64, use_tanh_limit=True)
X = torch.randn(8, 20)
y = torch.randint(0, 2, (8,))

# Forward с проверкой
h = model.embed(X)
print("embed:", h.abs().max().item(), "nan:", torch.isnan(h).any().item())

x_re, x_im = h, torch.zeros_like(h)

# restore
restore_re, restore_im = model.etvp.hyperbolic_restore(x_re, x_im)
print("restore:", restore_re.abs().max().item(), "nan:", torch.isnan(restore_re).any().item())

# complex_mul
f_re, f_im = model.etvp.complex_mul(x_re, x_im)
print("mul:", f_re.abs().max().item(), "nan:", torch.isnan(f_re).any().item())

# cardioid
f_re, f_im = model.etvp.cardioid_activation(f_re, f_im)
print("cardioid:", f_re.abs().max().item(), "nan:", torch.isnan(f_re).any().item())

# backward
out = model.head(f_re)
loss = F.cross_entropy(out, y)
print("loss:", loss.item(), "nan:", math.isnan(loss.item()))
loss.backward()

# проверить градиенты
for name, p in model.named_parameters():
    if p.grad is not None:
        g = p.grad.abs().max().item()
        if math.isnan(g) or g > 1e3:
            print(f"  ПЛОХО: {name} grad = {g}")
        else:
            print(f"  ок: {name} grad = {g:.4f}")
