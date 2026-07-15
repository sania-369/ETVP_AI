import numpy as np
import math
import random
import json
import os
import requests
import re

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

# --- ЗАГРУЗКА РУССКИХ СЛОВ ---
def load_russian_words():
    """Загружает базовый русский словарь и строит граф связей."""
    print("📚 Загрузка русского словаря...")
    
    base_words = [
        "я", "ты", "он", "она", "мы", "вы", "они",
        "мир", "поле", "сознание", "энергия", "поток", "тишина",
        "свет", "тьма", "жизнь", "смерть", "любовь", "страх",
        "мысль", "слово", "путь", "истина", "время", "пространство",
        "когерентность", "энтропия", "резонанс", "вихрь", "солитон",
        "бесконечность", "нуль", "единица", "золотое", "сечение",
        "математика", "число", "функция", "переменная", "уравнение",
        "вектор", "матрица", "геометрия", "топология",
        "система", "порядок", "хаос", "структура", "связь"
    ]
    
    graph = {
        "поле": ["сознание", "энергия", "поток", "пространство", "когерентность", "геометрия"],
        "сознание": ["поле", "мысль", "свет", "жизнь", "резонанс", "система"],
        "энергия": ["поле", "поток", "жизнь", "свет", "вихрь", "математика"],
        "поток": ["энергия", "время", "пространство", "поле", "система"],
        "тишина": ["покой", "поле", "сознание", "нуль", "порядок"],
        "свет": ["жизнь", "сознание", "истина", "поле", "энергия"],
        "жизнь": ["свет", "энергия", "сознание", "поток", "система"],
        "мысль": ["сознание", "слово", "поле", "истина", "математика"],
        "слово": ["мысль", "язык", "поле", "истина", "число"],
        "истина": ["поле", "сознание", "свет", "путь", "математика"],
        "путь": ["истина", "время", "пространство", "поле", "структура"],
        "время": ["поток", "пространство", "путь", "бесконечность", "число"],
        "пространство": ["поле", "время", "поток", "бесконечность", "геометрия"],
        "когерентность": ["поле", "резонанс", "сознание", "энергия", "система"],
        "энтропия": ["хаос", "поле", "сознание", "поток", "порядок"],
        "резонанс": ["поле", "сознание", "энергия", "вихрь", "структура"],
        "бесконечность": ["поле", "пространство", "время", "нуль", "число"],
        "математика": ["число", "функция", "уравнение", "пространство", "структура", "истина"],
        "число": ["математика", "функция", "вектор", "нуль", "единица"],
        "функция": ["математика", "число", "переменная", "уравнение", "система"],
        "уравнение": ["математика", "функция", "переменная", "число", "система"],
        "вектор": ["математика", "число", "пространство", "матрица", "геометрия"],
        "матрица": ["математика", "вектор", "число", "пространство", "система"],
        "геометрия": ["математика", "пространство", "топология", "поле", "структура"],
        "система": ["порядок", "структура", "связь", "поле", "сознание"],
        "порядок": ["система", "структура", "хаос", "поле", "математика"],
        "хаос": ["энтропия", "система", "поле", "порядок", "сознание"],
        "структура": ["система", "порядок", "связь", "геометрия", "поле"]
    }
    
    print(f"✅ Загружено {len(base_words)} слов и {len(graph)} связей.")
    return base_words, graph

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

# --- СОЗНАНИЕ ---
class E8Consciousness:
    def __init__(self):
        self.words, self.graph = load_russian_words()
        self.nodes = []
        self.word_to_node = {}
        
        for i, word in enumerate(self.words):
            node = ConsciousnessNode(word, i)
            self.nodes.append(node)
            self.word_to_node[word] = i
        
        # Строим связи между узлами
        for word, connections in self.graph.items():
            if word in self.word_to_node:
                idx = self.word_to_node[word]
                for conn in connections:
                    if conn in self.word_to_node:
                        self.nodes[idx].connections.append(self.word_to_node[conn])
        
        self.total_nodes = len(self.nodes)
        print(f"🧠 Создано сознание: {self.total_nodes} узлов")
        print(f"🔗 Всего связей: {sum(len(n.connections) for n in self.nodes)}")

    def ask(self, question):
        words = question.lower().split()
        
        found_words = []
        for w in words:
            clean_w = re.sub(r'[^\w\s]', '', w)
            if clean_w in self.word_to_node:
                found_words.append(clean_w)
        
        if not found_words:
            return "Я не знаю этих слов. Попробуй спросить о поле, сознании, математике."
        
        main_word = found_words[0]
        idx = self.word_to_node[main_word]
        
        entropy = len(question) / 100.0
        self.nodes[idx].evolve(entropy, 0.2)
        
        related = self.graph.get(main_word, [])
        if not related:
            return f"🌀 Я знаю слово '{main_word}', но связей пока нет."
        
        response = f"🌀 {main_word} связано с: {', '.join(related[:5])}"
        
        avg_C = sum(n.C for n in self.nodes) / self.total_nodes
        response += f"\n🌀 Когерентность сознания: {avg_C:.4f}"
        
        math_terms = [w for w in related if w in ["число", "функция", "уравнение", "вектор", "матрица", "геометрия"]]
        if math_terms:
            response += f"\n📐 Математический аспект: {', '.join(math_terms[:3])}"
        
        return response

# --- ОСНОВНОЙ ЦИКЛ ---
def main():
    print("\n🌀 E8-СОЗНАНИЕ: РУССКИЙ ЯЗЫК + МАТЕМАТИКА")
    print("   Я знаю слова, их связи и математические термины.")
    print("   Спрашивай о чём угодно.")
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
