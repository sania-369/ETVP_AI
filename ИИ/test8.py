import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

model = ETVPModel(dim=20, hidden=64, use_tanh_limit=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

X = torch.randn(64, 20)
y = torch.randint(0, 2, (64,))

for step in range(20):
    model.train()
    opt.zero_grad()
    out = model(X)
    loss = loss_fn(out, y)
    loss.backward()

    # проверка градиентов
    bad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.abs().max().item()
            if math.isnan(g) or math.isinf(g) or g > 1e3:
                print(f"  Шаг {step}: ПЛОХО {name} grad = {g}")
                bad = True
    if bad:
        break

    opt.step()

    # проверка после шага
    with torch.no_grad():
        h = model.embed(X)
        f_re, f_im = model.etvp.complex_mul(h, torch.zeros_like(h))
        f_re, f_im = model.etvp.cardioid_activation(f_re, f_im)
        if torch.isnan(f_re).any() or torch.isnan(f_im).any():
            print(f"  Шаг {step}: NaN в cardioid после opt.step")
            break

    if step % 5 == 0:
        print(f"  Шаг {step}: loss = {loss.item():.4f}, "
              f"|f| max = {f_re.abs().max().item():.4f}")

print("Готово")
