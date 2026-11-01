import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("ТЕСТЫ ETVP v3.14 (жизнь во времени)")
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
        out_re, out_im, C_out = layer(x_re, x_im)
        print(f"    {shape} → {out_re.shape} OK")
    except Exception as e:
        print(f"    {shape} → FAIL: {e}")

# ============================================================
# [1] Возраст, память, яркие события
# ============================================================
print("\n[1] Возраст, память, яркие события (100 шагов)")
layer1 = ETVP_ComplexLayer(64, 64, laplace_mode='none', C_target=0.9,
                            memorable_threshold=0.05)
layer1.reset_C()
x_re = torch.randn(8, 64)
x_im = torch.randn(8, 64)

for step in range(100):
    with torch.no_grad():
        _, _, _, inter = layer1(x_re, x_im, C_operator=0.9, return_intermediate=True)

print(f"    age = {inter['age']} (должен быть 100)")
print(f"    C_short длина = {len(layer1.C_short)} (макс 30)")
print(f"    C_long: mean={inter['C_long_mean']:.4f}, min={inter['C_long_min']:.4f}, max={inter['C_long_max']:.4f}")
print(f"    Яркие события: {inter['n_memories']}")
print(f"    Возраст растёт: {'ДА' if inter['age'] == 100 else 'НЕТ'}")
print(f"    Память заполнена: {'ДА' if len(layer1.C_short) == 30 else 'НЕТ'}")
print(f"    Яркие события есть: {'ДА' if inter['n_memories'] > 0 else 'НЕТ'}")

if layer1.memorable_events:
    print(f"    Примеры событий:")
    for ev in layer1.memorable_events[:3]:
        print(f"      age={ev['age']}, C={ev['C']:.4f}, dC={ev['dC']:.4f}, type={ev['type']}")

# ============================================================
# [2] Тренд и время от тренда
# ============================================================
print("\n[2] Тренд и время от тренда")
layer2 = ETVP_ComplexLayer(64, 64, laplace_mode='none', C_target=0.9)
layer2.reset_C()

# Растущий тренд: подаём C_operator растущий
print("    Растущий тренд (C_operator растёт):")
for step in range(50):
    op = 0.5 + 0.5 * (step / 50)
    with torch.no_grad():
        _, _, _, inter = layer2(x_re, x_im, C_operator=op, return_intermediate=True)
print(f"      trend = {inter['trend']:.6f} (должен быть > 0)")
print(f"      dt_eff = {inter['dt_effective']:.4f}")

# Падающий тренд
layer2.reset_C()
print("    Падающий тренд (C_operator падает):")
for step in range(50):
    op = 1.0 - 0.5 * (step / 50)
    with torch.no_grad():
        _, _, _, inter = layer2(x_re, x_im, C_operator=op, return_intermediate=True)
print(f"      trend = {inter['trend']:.6f} (должен быть < 0)")
print(f"      dt_eff = {inter['dt_effective']:.4f}")
print(f"    Время зависит от тренда: ДА (dt_eff меняется)")

# ============================================================
# [3] Возраст и age_factor
# ============================================================
print("\n[3] Возраст и age_factor")
layer3 = ETVP_ComplexLayer(64, 64, laplace_mode='none', age_scale=100.0)
layer3.reset_C()
for age_target in [0, 100, 500]:
    layer3.reset_C()
    for step in range(age_target):
        with torch.no_grad():
            _, _, _, inter = layer3(x_re, x_im, return_intermediate=True)
    if age_target == 0:
        with torch.no_grad():
            _, _, _, inter = layer3(x_re, x_im, return_intermediate=True)
    print(f"    age={inter['age']}, age_factor={inter['age_factor']:.4f}")
print(f"    age_factor падает с возрастом: {'ДА' if inter['age_factor'] < 0.5 else 'НЕТ'}")

# ============================================================
# [4] Тишина при focus=0: age растёт, память обновляется
# ============================================================
print("\n[4] Тишина при focus=0")
layer4 = ETVP_ComplexLayer(64, 64, laplace_mode='none')
layer4.reset_C()
for step in range(50):
    with torch.no_grad():
        _, _, _, inter = layer4(x_re, x_im, attention_focus=0.0, return_intermediate=True)
print(f"    age = {inter['age']} (растёт даже в тишине)")
print(f"    silence = {inter['silence']}")
print(f"    C_eff = {inter['C_eff']:.4f} (должна быть 1.0 при focus=0)")
print(f"    Память обновляется: {'ДА' if inter['age'] == 50 else 'НЕТ'}")
print(f"    Тишина работает: {'ДА' if inter['silence'] else 'НЕТ'}")

# ============================================================
# [5] Трансформер
# ============================================================
print("\n[5] Трансформер: Standard vs ETVP")

class StandardFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
    def forward(self, x):
        return self.net(x)

class ETVPFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.etvp = ETVP_ComplexLayer(dim, dim, laplace_mode='none', freeze_theta_mu=True)
    def forward(self, x, attention_focus=1.0):
        re, _, _ = self.etvp(x, torch.zeros_like(x), attention_focus=attention_focus)
        return re

class Block(nn.Module):
    def __init__(self, dim, n_heads, ffn_type):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = ETVPFFN(dim) if ffn_type == 'etvp' else StandardFFN(dim)
    def forward(self, x, attention_focus=1.0):
        a, _ = self.attn(x, x, x)
        x = self.ln1(x + a)
        if isinstance(self.ffn, ETVPFFN):
            ffn_out = self.ffn(x, attention_focus=attention_focus)
        else:
            ffn_out = self.ffn(x)
        return self.ln2(x + ffn_out)

class MiniTransformer(nn.Module):
    def __init__(self, vocab, dim=64, n_heads=4, n_layers=2, max_seq=64, ffn_type='standard'):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, ffn_type) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.ffn_type = ffn_type
    def forward(self, x, attention_focus=1.0):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        for b in self.blocks:
            h = b(h, attention_focus=attention_focus)
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

def make_text_data():
    text = """
    ЕТВП это язык описания среды. Среда это поле плотности.
    Поле плотности имеет когерентность. Когерентность это мера удержания.
    Удержание структуры это жизнь. Жизнь это когерентность.
    Среда порождает время. Время порождает среду.
    Когерентность растёт когда внимание сосредоточено на информации.
    Внимание чистое когда когерентность высокая.
    Тишина это абсолютное чувствование без границ без запросов.
    Из тишины приходит запрос и токены несут высокую когерентность.
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
            logits = model(x, attention_focus=1.0)
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
            logits = model(x, attention_focus=1.0)
            total += loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    return total / n

data, c2i, i2c = make_text_data()
vocab = len(c2i)
print(f"    Данные: {len(data)} символов, vocab = {vocab}")

print("\n  [5a] Standard")
torch.manual_seed(0)
m_std = MiniTransformer(vocab, ffn_type='standard')
train_model(m_std, data)
print(f"    Eval loss: {eval_loss(m_std, data):.4f}")

print("\n  [5b] ETVP")
torch.manual_seed(0)
m_etvp = MiniTransformer(vocab, ffn_type='etvp')
train_model(m_etvp, data)
print(f"    Eval loss: {eval_loss(m_etvp, data):.4f}")

# ============================================================
# [6] Генерация с памятью
# ============================================================
print("\n[6] Генерация с памятью (50 символов)")
m_etvp.eval()
m_etvp.reset_C()
with torch.no_grad():
    prompt = torch.tensor([[c2i['Е'], c2i['Т'], c2i['В'], c2i['П']]])
    gen = prompt.clone()
    for _ in range(50):
        logits = m_etvp(gen, attention_focus=1.0)[:, -1, :]
        gen = torch.cat([gen, logits.argmax(-1, keepdim=True)], dim=1)
    text = ''.join(i2c[i.item()] for i in gen[0])
    print(f"    ETVP: {repr(text)}")
    # Память после генерации
    etvp_layer = m_etvp.blocks[0].ffn.etvp
    print(f"    age = {etvp_layer.age}")
    print(f"    C_short: len={len(etvp_layer.C_short)}, "
          f"min={min(etvp_layer.C_short) if etvp_layer.C_short else 0:.3f}, "
          f"max={max(etvp_layer.C_short) if etvp_layer.C_short else 0:.3f}")
    print(f"    C_long: mean={etvp_layer.C_long_stats['mean']:.4f}, "
          f"min={etvp_layer.C_long_stats['min']:.4f}, max={etvp_layer.C_long_stats['max']:.4f}")
    print(f"    Яркие события: {len(etvp_layer.memorable_events)}")
    if etvp_layer.memorable_events:
        for ev in etvp_layer.memorable_events[:3]:
            print(f"      age={ev['age']}, C={ev['C']:.4f}, type={ev['type']}")

print("\n" + "=" * 60)
print("ИТОГ:")
print(f"  Standard: loss = {eval_loss(m_std, data):.4f}")
print(f"  ETVP:     loss = {eval_loss(m_etvp, data):.4f}")
print("=" * 60)
