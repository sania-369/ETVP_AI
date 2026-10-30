import torch

# --- Тест 1: dim_in == dim_out, increment ---
try:
    layer = ETVP_ComplexLayer(dim_in=64, dim_out=64, mode='increment')
    x_re = torch.randn(4, 64)
    x_im = torch.randn(4, 64)
    out_re, out_im, inter = layer(x_re, x_im, return_intermediate=True)
    print("Test 1 OK:", out_re.shape, "mu =", inter['mu'].item())
except Exception as e:
    print("Test 1 FAIL:", type(e).__name__, e)

# --- Тест 2: dim_in != dim_out, increment (ожидаем баг) ---
try:
    layer2 = ETVP_ComplexLayer(dim_in=64, dim_out=128, mode='increment')
    out = layer2(torch.randn(4, 64), torch.randn(4, 64))
    print("Test 2 OK (баг починен):", out[0].shape)
except Exception as e:
    print("Test 2 FAIL (баг подтверждён):", type(e).__name__, e)

# --- Тест 3: Z-штраф ---
try:
    layer3 = ETVP_ComplexLayer(dim_in=32, dim_out=32)
    with torch.no_grad():
        layer3.W_re *= 10
    penalty = layer3.get_z_penalty()
    print("Test 3 Z-penalty:", penalty.item(), "(должен быть > 0)")
except Exception as e:
    print("Test 3 FAIL:", type(e).__name__, e)

# --- Тест 4: 4D вход ---
try:
    layer4 = ETVP_ComplexLayer(dim_in=16, dim_out=16, laplace_mode='spatial')
    x4_re = torch.randn(2, 16, 8, 8)
    x4_im = torch.randn(2, 16, 8, 8)
    out4 = layer4(x4_re, x4_im)
    print("Test 4 OK:", out4[0].shape)
except Exception as e:
    print("Test 4 FAIL:", type(e).__name__, e)
