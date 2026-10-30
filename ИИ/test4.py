import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# (вставь сюда класс ETVP_ComplexLayer из предыдущего сообщения)

torch.manual_seed(0)

# ============================================================
# ТЕСТ A: 1000 шагов эволюции без обучения
# ============================================================
print("=" * 60)
print("ТЕСТ A: 1000 шагов эволюции, без градиентов")
print("=" * 60)

layer = ETVP_ComplexLayer(64, 64, laplace_mode='none', R_max=2.0)
x_re = torch.randn(8, 64)
x_im = torch.randn(8, 64)

norms = []
for step in range(1000):
    with torch.no_grad():
        x_re, x_im = layer(x_re, x_im, dt=0.1)
    mag = torch.sqrt(x_re**2 + x_im**2).max().item()
    norms.append(mag)
    if math.isnan(mag) or math.isinf(mag):
        print(f"  Шаг {step}: ВЗРЫВ (mag = {mag})")
        break
    if step % 200 == 0:
        print(f"  Шаг {step}: max|z| = {mag:.4f}")

if not any(math.isnan(n) or math.isinf(n) for n in norms):
    print(f"  Итог: 1000 шагов, max|z| = {max(norms):.4f}, min = {min(norms):.4f}")
    print(f"  Стабильно: {'ДА' if max(norms) < 100 else 'НЕТ'}")

# ============================================================
# ТЕСТ C: 100 шагов backward
# ============================================================
print("\n" + "=" * 60)
print("ТЕСТ C: 100 шагов backward, проверка градиентов")
print("=" * 60)

layer2 = ETVP_ComplexLayer(64, 64, laplace_mode='none', R_max=2.0)
x_re = torch.randn(8, 64, requires_grad=True)
x_im = torch.randn(8, 64, requires_grad=True)

grad_norms = []
for step in range(100):
    z_re, z_im = layer2(x_re, x_im, dt=0.1)
    loss = (z_re**2 + z_im**2).mean()
    loss.backward(retain_graph=True)

    g = 0.0
    for p in layer2.parameters():
        if p.grad is not None:
            g += p.grad.norm().item() ** 2
    g = math.sqrt(g)
    grad_norms.append(g)

    if math.isnan(g) or math.isinf(g):
        print(f"  Шаг {step}: ГРАДИЕНТ ВЗОРВАЛСЯ (||g|| = {g})")
        break
    if step % 20 == 0:
        print(f"  Шаг {step}: ||grad|| = {g:.4f}")

if not any(math.isnan(n) or math.isinf(n) for n in grad_norms):
    print(f"  Итог: 100 шагов, ||grad|| max = {max(grad_norms):.4f}")
    print(f"  Стабильно: {'ДА' if max(grad_norms) < 1e3 else 'НЕТ'}")
