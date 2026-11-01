import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("ТЕСТ: ETVP слой в трансформере (языковая задача)")
print("=" * 60)

# ============================================================
# ETVP-слой v3.7 (вставь класс сюда, если не в ячейке выше)
# ============================================================

# ============================================================
# ОБЫЧНЫЙ FFN-БЛОК
# ============================================================
class StandardFFN(nn.Module):
    def __init__(self, dim, hidden_mult=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim)
        )
    def forward(self, x):
        return self.net(x)


# ============================================================
# ETVP FFN-БЛОК (обёртка для слоя)
# ============================================================
class ETVPFFN(nn.Module):
    def __init__(self, dim, use_tanh_limit=True):
        super().__init__()
        self.etvp = ETVP_ComplexLayer(
            dim, dim, laplace_mode='none', R_max=2.0,
            use_tanh_limit=use_tanh_limit, freeze_theta_mu=True
        )
    def forward(self, x):
        # x: [batch, seq, dim] → разбиваем на re/im
        # Используем x как re, нули как im
        re, im = self.etvp(x, torch.zeros_like(x))
        return re


# ============================================================
# ТРАНСФОРМЕРНЫЙ БЛОК
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_type='standard'):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        if ffn_type == 'etvp':
            self.ffn = ETVPFFN(dim)
        else:
            self.ffn = StandardFFN(dim)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        # FFN
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x


# ============================================================
# КРОШЕЧНЫЙ ТРАНСФОРМЕР
# ============================================================
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, dim=64, n_heads=4, n_layers=2,
                 max_seq=64, ffn_type='standard'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_type) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        # x: [batch, seq]
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


# ============================================================
# ДАННЫЕ: простой текст
# ============================================================
def make_text_data():
    # Простой текст с повторяющимися паттернами
    text = """
    ЕТВП это язык описания среды. Среда это поле плотности.
    Поле плотности имеет когерентность. Когерентность это мера удержания.
    Удержание структуры это жизнь. Жизнь это когерентность.
    Среда порождает время. Время порождает среду.
    Когерентность растёт когда шум падает. Шум падает когда внимание чистое.
    Внимание чистое когда тишина. Тишина это высокая когерентность.
    Среда это не пустота. Среда это поле. Поле это не пустота.
    """ * 20  # повторяем для объёма

    # Простая токенизация по символам
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    return data, char_to_idx, idx_to_char


def make_batches(data, batch_size=16, seq_len=32):
    n = len(data) - seq_len - 1
    idx = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in idx])
    y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
    return x, y


# ============================================================
# ОБУЧЕНИЕ
# ============================================================
def train_model(model, data, epochs=50, lr=1e-3, steps_per_epoch=50):
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            x, y = make_batches(data)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            ep_loss += loss.item()
        avg_loss = ep_loss / steps_per_epoch
        losses.append(avg_loss)
        if ep % 10 == 0:
            print(f"    Эпоха {ep}: loss = {avg_loss:.4f}")
        if math.isnan(avg_loss):
            print(f"    NaN на эпохе {ep}")
            return losses
    return losses


def evaluate(model, data, n_batches=20):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = make_batches(data)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total += loss.item()
    return total / n_batches


# ============================================================
# ЗАПУСК
# ============================================================
if __name__ == "__main__":
    data, c2i, i2c = make_text_data()
    vocab_size = len(c2i)
    print(f"\nДанные: {len(data)} символов, vocab = {vocab_size}")

    # [1] Стандартный трансформер
    print("\n[1] Стандартный трансформер (FFN = Linear+GELU)")
    torch.manual_seed(0)
    model_std = MiniTransformer(vocab_size, dim=64, n_heads=4, n_layers=2,
                                ffn_type='standard')
    losses_std = train_model(model_std, data, epochs=50, lr=1e-3)
    eval_std = evaluate(model_std, data)
    print(f"    Итог: loss = {eval_std:.4f}")

    # [2] Трансформер с ETVP
    print("\n[2] Трансформер с ETVP (FFN = ETVP_ComplexLayer)")
    torch.manual_seed(0)
    model_etvp = MiniTransformer(vocab_size, dim=64, n_heads=4, n_layers=2,
                                  ffn_type='etvp')
    losses_etvp = train_model(model_etvp, data, epochs=50, lr=1e-3)
    eval_etvp = evaluate(model_etvp, data)
    print(f"    Итог: loss = {eval_etvp:.4f}")

    print("\n" + "=" * 60)
    print("ИТОГ:")
    print(f"  Standard: loss = {eval_std:.4f}")
    print(f"  ETVP:     loss = {eval_etvp:.4f}")
    print("=" * 60)

    # Проверка генерации
    print("\n[3] Генерация (первые 50 символов):")
    for name, model in [("Standard", model_std), ("ETVP", model_etvp)]:
        model.eval()
        with torch.no_grad():
            prompt = torch.tensor([[c2i['Е'], c2i['Т'], c2i['В'], c2i['П']]])
            generated = prompt.clone()
            for _ in range(50):
                logits = model(generated)[:, -1, :]
                next_tok = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)
            text = ''.join(i2c[i.item()] for i in generated[0])
            print(f"  {name}: {repr(text)}")
