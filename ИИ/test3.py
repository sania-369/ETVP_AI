import torch
torch.manual_seed(0)

# Возьмём один батч
X = torch.randn(8, 20)
embed = torch.nn.Linear(20, 64)
h = embed(X)

layer = ETVP_ComplexLayer(64, 64, laplace_mode='none')
x_re, x_im = h, torch.zeros_like(h)

print("h:", h.abs().max().item(), "nan:", torch.isnan(h).any().item())

# Шаг 1: complex_mul
f_re, f_im = layer.complex_mul(x_re, x_im)
print("f после mul:", f_re.abs().max().item(), "nan:", torch.isnan(f_re).any().item())

# Шаг 2: cardioid
f_re, f_im = layer.cardioid_activation(f_re, f_im)
print("f после cardioid:", f_re.abs().max().item(), "nan:", torch.isnan(f_re).any().item())

# Шаг 3: mu
mu = torch.nn.functional.softplus(layer.theta_mu)
print("mu:", mu.item())

# Шаг 4: lap (должен быть 0 при laplace_mode='none')
lap_re = layer.laplacian(f_re)
print("lap:", lap_re.abs().max().item())

# Шаг 5: k1
k1_re = f_re - mu * lap_re
print("k1:", k1_re.abs().max().item(), "nan:", torch.isnan(k1_re).any().item())

# Шаг 6: x_pred
x_pred_re = x_re + 0.1 * k1_re
print("x_pred:", x_pred_re.abs().max().item(), "nan:", torch.isnan(x_pred_re).any().item())

# Шаг 7: k2
f_pred_re, f_pred_im = layer.complex_mul(x_pred_re, x_im)
f_pred_re, f_pred_im = layer.cardioid_activation(f_pred_re, f_pred_im)
k2_re = f_pred_re
print("k2:", k2_re.abs().max().item(), "nan:", torch.isnan(k2_re).any().item())

# Шаг 8: финал
z_re = x_re + 0.1 * (k1_re + k2_re) / 2
print("z:", z_re.abs().max().item(), "nan:", torch.isnan(z_re).any().item())
