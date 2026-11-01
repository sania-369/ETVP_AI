import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("ТЕСТЫ ETVP v3.9 (C как состояние)")
print("=" * 60)

# ============================================================
# [0] Проверка форм
# ============================================================
print("\n[0] Проверка форм")
layer = ETVP_ComplexLayer(16, 16, laplace_mode='none')
for shape in [(4, 16), (2, 16, 16), (2, 16, 8, 8)]:
    try:
        x_re = torch.randn(*shape)
        x_im = torch.randn(*shape)
        out_re, out_im = layer(x_re, x_im)
        print(f"    {shape} → {out_re.shape} OK")
    except Exception as e:
        print(f"    {shape} → FAIL: {e}")

# ============================================================
# [1] C релаксирует: 50 шагов, батч 64
# ============================================================
print("\n[1] C релаксирует: 50 шагов, батч 64")
class ETVPModel(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.embed = nn.Linear(dim, hidden)
        self.etvp = ETVP_ComplexLayer(hidden, hidden, laplace_mode='none',
                                       R_max=2.0, freeze_theta_mu=True,
                                       gamma_c=0.1, C_FFS=1.0)
        self.head = nn.Linear(hidden, 2)
    def forward(self, x):
        h = self.embed(x)
        re, _ = self.etvp(h, torch.zeros_like(h))
        return self.head(re)
    def get_param_groups(self, wd=1e-3):
        return [
            {'params': list(self.embed.parameters()), 'weight_decay': 0.0},
            *self.etvp.get_param_groups(wd),
            {'params': list(self.head.parameters()), 'weight_decay': 0.0},
        ]

torch.manual_seed(0)
model = ETVPModel(20, 64)
model.etvp.reset_C()
opt = torch.optim.Adam(model.get_param_groups(1e-3), lr=1e-2, eps=1e-4)
loss_fn = nn.CrossEntropyLoss()
X = torch.randn(64, 20)
y = torch.randint(0, 2, (64,))

nan_found = False
for step in range(50):
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    bad = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
              for p in model.parameters())
    if bad:
        print(f"    Шаг {step}: NaN в градиентах")
        nan_found = True
        break
    opt.step()

if not nan_found:
    C_hist = model.etvp.C_history
    print(f"    OK: 50 шагов, loss = {loss.item():.4f}")
    print(f"    C: start={C_hist[0]:.4f}, end={C_hist[-1]:.4f}, "
          f"min={min(C_hist):.4f}, max={max(C_hist):.4f}")

# ============================================================
# [2] Трансформер: Standard vs ETVP
# ============================================================
print("\n[2] Трансформер: Standard vs ETVP (3D)")

class StandardFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
    def forward(self, x):
        return self.net(x)

class ETVPFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.etvp = ETVP_ComplexLayer(dim, dim, laplace_mode='none',
                                       R_max=2.0, freeze_theta_mu=True,
                                       gamma_c=0.1)
    def forward(self, x):
        re, _ = self.etvp(x, torch.zeros_like(x))
        return re

class Block(nn.Module):
    def __init__(self, dim, n_heads, ffn_type):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = ETVPFFN(dim) if ffn_type == 'etvp' else StandardFFN(dim)
    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.ln1(x + a)
        return self.ln2(x + self.ffn(x))

class MiniTransformer(nn.Module):
    def __init__(self, vocab, dim=64, n_heads=4, n_layers=2, max_seq=64, ffn_type='standard'):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, ffn_type) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.ffn_type = ffn_type
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h))
    def get_param_groups(self, wd=1e-3):
        groups = [
            {'params': list(self.embed.parameters()), 'weight_decay': 0.0},
            {'params': list(self.pos.parameters()), 'weight_decay': 0.0},
        ]
        for b in self.blocks:
            groups.append({'params': list(b.attn.parameters()), 'weight_decay': 0.0})
            groups.append({'params': list(b.ln1.parameters()), 'weight_decay': 0.0})
            groups.append({'params': list(b.ln2.parameters()), 'weight_decay': 0.0})
            if self.ffn_type == 'etvp':
                groups.extend(b.ffn.etvp.get_param_groups(wd))
            else:
                groups.append({'params': list(b.ffn.parameters()), 'weight_decay': 0.0})
        groups.append({'params': list(self.ln_f.parameters()), 'weight_decay': 0.0})
        groups.append({'params': list(self.head.parameters()), 'weight_decay': 0.0})
        return groups
    def reset_C(self):
        for b in self.blocks:
            if self.ffn_type == 'etvp':
                b.ffn.etvp.reset_C()
    def get_C_history(self):
        if self.ffn_type != 'etvp':
            return []
        return self.blocks[0].ffn.etvp.C_history

def make_text_data():
    text = """
    ЕТВП это язык описания среды. Среда это поле плотности.
    Поле плотности имеет когерентность. Когерентность это мера удержания.
    Удержание структуры это жизнь. Жизнь это когерентность.
    Среда порождает время. Время порождает среду.
    Когерентность растёт когда шум падает. Шум падает когда внимание чистое.
    Внимание чистое когда тишина. Тишина это высокая когерентность.
    Среда это не пустота. Среда это поле. Поле это не пустота.
    """ * 20
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    data = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    return data, c2i, i2c

def make_batches(data, bs=16, sl=32):
    n = len(data) - sl - 1
    idx = torch.randint(0, n, (bs,))
    x = torch.stack([data[i:i+sl] for i in idx])
    y = torch.stack([data[i+1:i+sl+1] for i in idx])
    return x, y

def train_model(model, data, epochs=50, lr=1e-3, steps=50):
    opt = torch.optim.Adam(model.get_param_groups(1e-3), lr=lr, eps=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.reset_C()
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for _ in range(steps):
            x, y = make_batches(data)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            ep_loss += loss.item()
        if ep % 10 == 0:
            print(f"    Эпоха {ep}: loss = {ep_loss/steps:.4f}")
        if math.isnan(ep_loss):
            print(f"    NaN на эпохе {ep}")
            return
    print(f"    Готово")

def eval_loss(model, data, n=20):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    with torch.no_grad():
        for _ in range(n):
            x, y = make_batches(data)
            logits = model(x)
            total += loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    return total / n

data, c2i, i2c = make_text_data()
vocab = len(c2i)
print(f"    Данные: {len(data)} символов, vocab = {vocab}")

print("\n  [2a] Standard")
torch.manual_seed(0)
m_std = MiniTransformer(vocab, ffn_type='standard')
train_model(m_std, data)
print(f"    Eval loss: {eval_loss(m_std, data):.4f}")

print("\n  [2b] ETVP")
torch.manual_seed(0)
m_etvp = MiniTransformer(vocab, ffn_type='etvp')
train_model(m_etvp, data)
print(f"    Eval loss: {eval_loss(m_etvp, data):.4f}")

# ============================================================
# [3] Генерация с траекторией C
# ============================================================
print("\n[3] Генерация с траекторией C (50 символов)")
for name, model in [("Standard", m_std), ("ETVP", m_etvp)]:
    model.eval()
    if name == "ETVP":
        model.reset_C()
    with torch.no_grad():
        prompt = torch.tensor([[c2i['Е'], c2i['Т'], c2i['В'], c2i['П']]])
        gen = prompt.clone()
        for _ in range(50):
            logits = model(gen)[:, -1, :]
            gen = torch.cat([gen, logits.argmax(-1, keepdim=True)], dim=1)
        text = ''.join(i2c[i.item()] for i in gen[0])
        print(f"    {name}: {repr(text)}")
        if name == "ETVP":
            C_hist = model.get_C_history()
            if C_hist:
                print(f"    C при генерации: start={C_hist[0]:.3f}, "
                      f"end={C_hist[-1]:.3f}, min={min(C_hist):.3f}, max={max(C_hist):.3f}")

print("\n" + "=" * 60)
print("ИТОГ:")
print(f"  Standard: loss = {eval_loss(m_std, data):.4f}")
print(f"  ETVP:     loss = {eval_loss(m_etvp, data):.4f}")
print("=" * 60)
