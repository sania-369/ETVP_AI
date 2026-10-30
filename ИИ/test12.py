import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

# Модель как в тесте
embed = nn.Linear(20, 64)
layer = ETVP_ComplexLayer(64, 64, laplace_mode='none', freeze_theta_mu=True)
head = nn.Linear(64, 2)

opt = torch.optim.Adam(list(embed.parameters()) + list(layer.parameters()) + list(head.parameters()), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

X = torch.randn(64, 20)
y = torch.randint(0, 2, (64,))

for step in range(50):
    opt.zero_grad()
    h = embed(X)
    
    # --- Пошаговая проверка внутри ETVP ---
    x_re, x_im = h, torch.zeros_like(h)
    
    restore_re, restore_im = layer.hyperbolic_restore(x_re, x_im)
    if torch.isnan(restore_re).any():
        print(f"Шаг {step}: NaN в restore"); break
    
    f_re, f_im = layer.complex_mul(x_re, x_im)
    if torch.isnan(f_re).any():
        print(f"Шаг {step}: NaN в complex_mul (k1)"); break
    
    f_re, f_im = layer.cardioid_activation(f_re, f_im)
    if torch.isnan(f_re).any():
        print(f"Шаг {step}: NaN в cardioid (k1)"); break
    
    mu = F.softplus(layer.theta_mu)
    f_abs = torch.sqrt(f_re**2 + f_im**2) + layer.EPS
    f_mag = torch.sqrt(f_re**2 + f_im**2 + 1e-6)
    noise_scale = layer.eta * torch.tanh(layer.beta * f_abs)
    noise_re = math.sqrt(0.1) * noise_scale * torch.randn_like(f_re) * (f_re / f_mag)
    if torch.isnan(noise_re).any():
        print(f"Шаг {step}: NaN в noise"); break
    
    k1_re = f_re - mu * layer.laplacian(f_re)
    x_pred_re = x_re + 0.1 * k1_re
    
    f_pred_re, f_pred_im = layer.complex_mul(x_pred_re, x_im)
    if torch.isnan(f_pred_re).any():
        print(f"Шаг {step}: NaN в complex_mul (k2)"); break
    
    f_pred_re, f_pred_im = layer.cardioid_activation(f_pred_re, f_pred_im)
    k2_re = f_pred_re - mu * layer.laplacian(f_pred_re)
    
    z_re = x_re + 0.1 * (k1_re + k2_re) / 2 + 0.1 * restore_re + noise_re
    if torch.isnan(z_re).any():
        print(f"Шаг {step}: NaN в z_next"); break
    
    z_re = layer.R_max * torch.tanh(z_re / layer.R_max)
    
    # --- Полный forward ---
    z_re_full, z_im_full = layer(h, torch.zeros_like(h))
    out = head(z_re_full)
    loss = loss_fn(out, y)
    loss.backward()
    
    # Проверить градиенты
    for name, p in list(embed.named_parameters()) + list(layer.named_parameters()) + list(head.named_parameters()):
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"Шаг {step}: NaN в градиенте {name}")
            break
    else:
        opt.step()
        if step % 10 == 0:
            print(f"Шаг {step}: ок, loss = {loss.item():.4f}")
        continue
    break
