import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)

def test_config(name, use_cardioid=True, use_tanh_limit=True, gamma_restore=0.1):
    embed = nn.Linear(20, 64)
    layer = ETVP_ComplexLayer(
        64, 64, laplace_mode='none', freeze_theta_mu=True,
        use_cardioid=use_cardioid, use_tanh_limit=use_tanh_limit,
        gamma_restore=gamma_restore
    )
    head = nn.Linear(64, 2)
    opt = torch.optim.Adam(
        list(embed.parameters()) + list(layer.parameters()) + list(head.parameters()),
        lr=1e-2
    )
    loss_fn = nn.CrossEntropyLoss()
    X = torch.randn(64, 20)
    y = torch.randint(0, 2, (64,))

    for step in range(50):
        opt.zero_grad()
        h = embed(X)
        z_re, z_im = layer(h, torch.zeros_like(h))
        loss = loss_fn(head(z_re), y)
        loss.backward()
        bad = any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in list(embed.parameters()) + list(layer.parameters()) + list(head.parameters())
        )
        if bad:
            print(f"  {name}: NaN на шаге {step}")
            return
        opt.step()
    print(f"  {name}: OK (50 шагов)")

print("Проверка компонентов:")
test_config("всё включено")
test_config("cardioid OFF", use_cardioid=False)
test_config("tanh_limit OFF", use_tanh_limit=False)
test_config("restore OFF", gamma_restore=0.0)
test_config("cardioid + tanh OFF", use_cardioid=False, use_tanh_limit=False)
