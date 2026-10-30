# test_etvp_layer.py
import torch
from etvp_layer import ETVP_ComplexLayer  # твой файл

# Тест 1: dim_in == dim_out, increment
layer = ETVP_ComplexLayer(dim_in=64, dim_out=64, mode='increment')
x_re = torch.randn(4, 64)
x_im = torch.randn(4, 64)
out_re, out_im, inter = layer(x_re, x_im, return_intermediate=True)
print("Test 1 OK:", out_re.shape, inter['mu'].item())

# Тест 2: dim_in != dim_out, increment (должен упасть, если баг не починен)
try:
    layer2 = ETVP_ComplexLayer(dim_in=64, dim_out=128, mode='increment')
    out = layer2(torch.randn(4, 64), torch.randn(4, 64))
    print("Test 2 OK (баг починен):", out[0].shape)
except RuntimeError as e:
    print("Test 2 FAIL (баг подтверждён):", e)

# Тест 3: Z-штраф
layer3 = ETVP_ComplexLayer(dim_in=32, dim_out=32)
with torch.no_grad():
    layer3.W_re *= 10  # искусственно раздуваем веса
penalty = layer3.get_z_penalty()
print("Test 3 Z-penalty:", penalty.item(), "(должен быть > 0)")

# Тест 4: 4D вход
layer4 = ETVP_ComplexLayer(dim_in=16, dim_out=16, laplace_mode='spatial')
x4_re = torch.randn(2, 16, 8, 8)
x4_im = torch.randn(2, 16, 8, 8)
out4 = layer4(x4_re, x4_im)
print("Test 4 OK:", out4[0].shape)
