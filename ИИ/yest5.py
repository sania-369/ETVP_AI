import torch
import math

print("=" * 60)
print("ТЕСТЫ ETVP Complex Layer v3.3")
print("=" * 60)

# --- Проверка версии ---
print("\n[0] Проверка версии класса")
print("    has hyperbolic_restore:", hasattr(ETVP_ComplexLayer, 'hyperbolic_restore'))
print("    has k_hyper param:", 'k_hyper' in ETVP_ComplexLayer.__init__.__code__.co_varnames)
print("    has use_tanh_limit:", 'use_tanh_limit' in ETVP_ComplexLayer.__init__.__code__.co_varnames)

# ============================================================
# Test 1: dim_in == dim_out, increment
# ============================================================
print("\n[1] dim_in=64, dim_out=64, increment")
try:
    layer1 = ETVP_ComplexLayer(dim_in=64, dim_out=64, mode='increment')
    x_re = torch.randn(4, 64)
    x_im = torch.randn(4, 64)
    out_re, out_im, inter = layer1(x_re, x_im, return_intermediate=True)
    print(f"    OK: out shape = {out_re.shape}")
    print(f"    mu = {inter['mu']}")
    print(f"    C = {inter['C']}")
    print(f"    restore max = {inter['restore'][0].abs().max().item():.4f}")
except Exception as e:
    print(f"    FAIL: {type(e).__name__}: {e}")

# ============================================================
# Test 2: dim_in != dim_out — ожидаем ValueError
# ============================================================
print("\n[2] dim_in=64, dim_out=128 (ожидаем ValueError)")
try:
    layer2 = ETVP_ComplexLayer(dim_in=64, dim_out=128)
    print("    FAIL: должно было упасть, но не упало")
except ValueError as e:
    print(f"    OK: правильная ошибка — {str(e)[:70]}")
except Exception as e:
    print(f"    FAIL: неверный тип ошибки — {type(e).__name__}: {e}")

# ============================================================
# Test 3: Z-штраф
# ============================================================
print("\n[3] Z-штраф при раздутых весах")
try:
    layer3 = ETVP_ComplexLayer(dim_in=32, dim_out=32)
    with torch.no_grad():
        layer3.W_re *= 10
        layer3.W_im *= 10
    penalty = layer3.get_z_penalty()
    print(f"    OK: penalty = {penalty.item():.4f} (должен быть > 0)")
    if penalty.item() <= 0:
        print("    ВНИМАНИЕ: penalty <= 0, Z-штраф не работает")
except Exception as e:
    print(f"    FAIL: {type(e).__name__}: {e}")

# ============================================================
# Test 4: 4D вход (2, 16, 8, 8)
# ============================================================
print("\n[4] 4D вход (2, 16, 8, 8), laplace_mode='spatial'")
try:
    layer4 = ETVP_ComplexLayer(dim_in=16, dim_out=16, laplace_mode='spatial')
    x4_re = torch.randn(2, 16, 8, 8)
    x4_im = torch.randn(2, 16, 8, 8)
    print(f"    Вход: {x4_re.shape}")
    out4 = layer4(x4_re, x4_im)
    print(f"    OK: out shape = {out4[0].shape}")
except Exception as e:
    print(f"    FAIL: {type(e).__name__}: {e}")

# ============================================================
# Test 5: Стабильность — 1000 шагов без обучения
# ============================================================
print("\n[5] Стабильность: 1000 шагов эволюции без обучения")
try:
    layer5 = ETVP_ComplexLayer(64, 64, laplace_mode='none', R_max=2.0)
    x_re = torch.randn(8, 64)
    x_im = torch.randn(8, 64)
    mags = []
    exploded = False
    for step in range(1000):
        with torch.no_grad():
            x_re, x_im = layer5(x_re, x_im, dt=0.1)
        mag = torch.sqrt(x_re**2 + x_im**2).max().item()
        mags.append(mag)
        if math.isnan(mag) or math.isinf(mag):
            print(f"    ВЗРЫВ на шаге {step}: mag = {mag}")
            exploded = True
            break
    if not exploded:
        print(f"    OK: 1000 шагов, max|z| = {max(mags):.4f}, min = {min(mags):.4f}")
        print(f"    Стабильно: {'ДА' if max(mags) < 10 else 'НЕТ (слишком большой рост)'}")
except Exception as e:
    print(f"    FAIL: {type(e).__name__}: {e}")

# ============================================================
# Test 6: Градиенты — 100 шагов backward
# ============================================================
print("\n[6] Градиенты: 100 шагов backward")
try:
    layer6 = ETVP_ComplexLayer(64, 64, laplace_mode='none', R_max=2.0)
    x_re = torch.randn(8, 64, requires_grad=True)
    x_im = torch.randn(8, 64, requires_grad=True)
    grad_norms = []
    for step in range(100):
        z_re, z_im = layer6(x_re, x_im, dt=0.1)
        loss = (z_re**2 + z_im**2).mean()
        loss.backward(retain_graph=True)
        g = 0.0
        for p in layer6.parameters():
            if p.grad is not None:
                g += p.grad.norm().item() ** 2
        g = math.sqrt(g)
        grad_norms.append(g)
        if math.isnan(g) or math.isinf(g):
            print(f"    ВЗРЫВ градиента на шаге {step}: ||g|| = {g}")
            break
    if not any(math.isnan(n) or math.isinf(n) for n in grad_norms):
        print(f"    OK: 100 шагов, ||grad|| max = {max(grad_norms):.4f}")
        print(f"    Стабильно: {'ДА' if max(grad_norms) < 1e3 else 'НЕТ'}")
except Exception as e:
    print(f"    FAIL: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("ТЕСТЫ ЗАВЕРШЕНЫ")
print("=" * 60)
