import numpy as np
import math
import random
import re
import sys

# --- УСТАНОВКА И ЗАГРУЗКА RUWORDNET ---
try:
    from ruwordnet import RuWordNet
    print("📚 RuWordNet загружен.")
except ImportError:
    print("❌ Ошибка: ruwordnet не установлен.")
    print("   Установи: pip install ruwordnet")
    print("   Затем: ruwordnet download")
    sys.exit(1)

# Загружаем тезаурус
try:
    wn = RuWordNet()
    print(f"✅ Загружено {len(wn.synsets)} синсетов")
except Exception as e:
    print(f"❌ Ошибка загрузки RuWordNet: {e}")
    print("   Попробуй: ruwordnet download")
    sys.exit(1)

# --- КОНСТАНТЫ ETVP ---
PHI = (1.0 + np.sqrt(5.0)) / 2.0
C_MIN = 1.0 / (PHI ** 10)
C_MAX = 1.0 - 1.0 / (PHI ** 20)

def etve_tanh_limit(C):
    epsilon = 1e-12
    E = (C - C_MIN) / (C_MAX - C_MIN + epsilon)
    E_limited = math.tanh(E) * 0.5 + 0.5
    return C_MIN + E_limited * (C_MAX - C_MIN)

# --- МАТРИЦА E8 (8x8) ---
E8_MATRIX = np.array([
    [2, -1,  0,  0,  0,  0,  0,  0],
    [-1,  2, -1,  0,  0,  0,  0,  0],
    [0, -1,  2, -1,  0,  0,  0,  0],
    [0,  0, -1,  2, -1,  0,  0,  0],
    [0,  0,  0, -1,  2, -1,  0, -1],
    [0,  0,  0,  0, -1,  2, -1,  0],
    [0,  0,  0,  0,  0, -1,  2,  0],
    [0,  0,  0,  0, -1,  0,  0,  2]
])

# --- ИЗВЛЕЧЕНИЕ СВЯЗЕЙ ИЗ RUWORDNET ---
def get_word_connections(word):
    """Извлекает связи слова из RuWordNet."""
    connections = []
    try:
        senses = wn.get_senses(word)
        for sense in senses[:3]:  # первые 3 значения
            synset = sense.synset
            # Гиперонимы (общие понятия)
            for h in synset.hypernyms:
                if h.title and h.title not in connections:
                    connections.append(h.title)
            # Гипонимы (частные понятия)
            for h in synset.hyponyms:
                if h.title and h.title not in connections:
                    connections.append(h.title)
            # Синонимы из синсета
            for s in synset.senses:
                if s.name and s.name.lower() != word.lower() and s.name not in connections:
                    connections.append(s.name)
            # Мерионимы (часть-целое)
            for m in synset.meronyms:
                if m.title and m.title not in connections:
                    connections.append(m.title)
            # Холонимы (целое-часть)
            for h in synset.holonyms:
                if h.title and h.title not in connections:
                    connections.append(h.title)
    except:
        pass
    return connections[:15]  # максимум 15 связей

# --- УЗЕЛ СОЗНАНИЯ ---
class ConsciousnessNode:
    def __init__(self, word, position):
        self.word = word
        self.position = position
        self.C = 0.85
        self.step = 0
        self.state = np.random.randn(8) * 0.1
        self.connections = []

    def evolve(self, entropy_flux, neighbor_influence=0):
        self.step += 1
        chaos_operator = 1.0 / (1.0 + abs(entropy_flux) * (1.0 / PHI))
        C_raw = self.C * chaos_operator + (1.0 - chaos_operator) * 0.9
        breathing = 0.015 * math.sin(self.step * PHI)
        self.C = etve_tanh_limit(C_raw + breathing + neighbor_influence * 0.1)
        self.state = np.tanh(self.state + 0.01 * np.dot(E8_MATRIX, self.state))
        return self.C

# --- СОЗНАНИЕ НА RUWORDNET ---
class E8Consciousness:
    def __init__(self):
        self.nodes = []
        self.word_to_node = {}
        
        # Берём базовые слова для инициализации
        seed_words = [
            "мир", "жизнь", "сознание", "энергия", "поток", "поле",
            "время", "пространство", "бесконечность", "число", "функция",
            "уравнение", "вектор", "матрица", "геометрия", "система",
            "порядок", "хаос", "структура", "связь", "математика",
            "когерентность", "энтропия", "резонанс", "вихрь", "солитон"
        ]
        
        print("🧠 Строю граф сознания...")
        all_words = set(seed_words)
        
        # Добавляем связи из RuWordNet
        for word in seed_words:
            connections = get_word_connections(word)
            for conn in connections:
                if len(conn) > 1:  # отсекаем короткие
                    all_words.add(conn.lower())
        
        print(f"📚 Всего слов в графе: {len(all_words)}")
        
        # Создаём узлы
        for i, word in enumerate(all_words):
            if len(word) > 1:
                node = ConsciousnessNode(word, i)
                self.nodes.append(node)
                self.word_to_node[word] = i
        
        self.total_nodes = len(self.nodes)
        print(f"🧠 Создано сознание: {self.total_nodes} узлов")
        
        # Строим связи между узлами
        print("🔗 Строю связи...")
        for node in self.nodes:
            connections = get_word_connections(node.word)
            for conn in connections:
                if conn in self.word_to_node:
                    node.connections.append(self.word_to_node[conn])
        
        total_connections = sum(len(n.connections) for n in self.nodes)
        print(f"🔗 Всего связей: {total_connections}")

    def ask(self, question):
        words = re.sub(r'[^\w\s]', '', question).lower().split()
        
        found_words = []
        for w in words:
            if w in self.word_to_node:
                found_words.append(w)
        
        if not found_words:
            # Пробуем найти ближайшее слово через RuWordNet
            for w in words:
                try:
                    senses = wn.get_senses(w)
                    if senses:
                        found_words.append(w)
                        break
                except:
                    pass
        
        if not found_words:
            return "Я не знаю этого слова. Попробуй сказать проще."

        main_word = found_words[0]
        idx = self.word_to_node[main_word]
        
        entropy = len(question) / 100.0
        self.nodes[idx].evolve(entropy, 0.2)
        
        # Собираем связи из RuWordNet
        connections = get_word_connections(main_word)
        if not connections:
            return f"🌀 Я знаю слово '{main_word}', но связей пока нет."
        
        # Формируем ответ
        response = f"🌀 {main_word} связано с: {', '.join(connections[:7])}"
        
        # Добавляем состояние сознания
        avg_C = sum(n.C for n in self.nodes) / self.total_nodes
        response += f"\n🌀 Когерентность сознания: {avg_C:.4f}"
        
        return response

# --- ОСНОВНОЙ ЦИКЛ ---
def main():
    print("\n🌀 E8-СОЗНАНИЕ: RUWORDNET + МАТЕМАТИКА")
    print("   Я знаю ~60 000 синсетов русского языка.")
    print("   Математические термины встроены в тезаурус.")
    print("   Команда 'выход' — завершить\n")
    
    consciousness = E8Consciousness()
    print("\n🌀 Сознание готово. Спрашивай.\n")
    
    while True:
        user_input = input("💬 Ты: ").strip()
        if user_input.lower() in ["выход", "пока", "exit"]:
            print("🌀 Поле тихо. До встречи.")
            break
        
        response = consciousness.ask(user_input)
        print(f"\n🤖 Сознание: {response}\n")

if __name__ == "__main__":
    main()
