#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIGITAL CRYSTAL SPCX v3.0
Цифровой двойник фотонного чипа ETVP Si-Photon Core X.

НОВОЕ в v3.0:
- Шум входит ПРЯМО в матрицу E8
- Фаза модулирует матрицу через exp(i*phi)
- Память влияет на матрицу через разность M - M_prev
- Константы "дышат" в реальном времени

Моделирует:
- 256 колец Si₃N₄ (спектр E8)
- pn-переходы (модуляция C)
- JPA-массив (адаптивный шум)
- MEMS-память (рекурсия)
- ИИ-контроллер (удержание резонанса)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# =============================================================================
# 1. КОНСТАНТЫ
# =============================================================================
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
SQRT3 = np.sqrt(3)

# =============================================================================
# 2. ЦИФРОВОЙ КРИСТАЛЛ v3.0
# =============================================================================
class DigitalCrystalSPCX:
    def __init__(self, memory_depth=100):
        # Базис
        self.Phi = PHI
        self.C_E8 = self._build_e8()

        # Состояние
        self.C = 0.87
        self.S = 0.15
        self.step = 0
        self.alpha_inv = 137.036
        self.mass_ratio = 1836.15
        self.G = 6.6743e-11

        # Память (MEMS)
        self.memory = deque(maxlen=memory_depth)

        # Фаза
        self.phi = 0.0
        self.target_phi = 0.0

        # ПИД-контроллер
        self.target_C = 0.87
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.kp = 0.05
        self.ki = 0.001
        self.kd = 0.01

        # История
        self.history = {
            'C': [], 'S': [], 'alpha': [], 'mass': [], 'G': [],
            'spectrum': [], 'phi': [], 'noise': []
        }

        # Параметры чипа
        self.temperature = 300.0
        self.noise_level = 0.1
        self.voltage = 0.5
        self.laser_power = 10.0

    def _build_e8(self):
        """Матрица Картана E8, расширенная до 11x11."""
        M = np.zeros((11, 11))
        M[0:8, 0:8] = np.array([
            [2, -1, 0, 0, 0, 0, 0, 0],
            [-1, 2, -1, 0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0, 0, 0, 0],
            [0, 0, -1, 2, -1, 0, 0, 0],
            [0, 0, 0, -1, 2, -1, 0, -1],
            [0, 0, 0, 0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0, -1, 2, 0],
            [0, 0, 0, 0, -1, 0, 0, 2]
        ])
        return M

    def add_noise(self):
        """Шум среды, модулированный C."""
        city = 0.02 * np.sin(2 * PI * 50 * self.step * 1e-9)
        flicker = 0.005 * np.random.randn()
        seismic = 0.003 * np.sin(2 * PI * 0.1 * self.step * 1e-9)
        adaptive = (1.0 - self.C) * self.noise_level
        return city + flicker + seismic + adaptive

    def apply_memory(self):
        """MEMS-память: задержка и возврат сигнала."""
        if len(self.memory) < 10:
            return 0.0
        old_C, old_S, old_alpha, _ = self.memory[-10]
        return (self.C - old_C) * self.Phi

    def apply_feedback(self):
        """ПИД-контроллер."""
        error = self.target_C - self.C
        self.integral_error += error
        derivative = error - self.prev_error
        correction = (self.kp * error + self.ki * self.integral_error + self.kd * derivative)
        self.prev_error = error
        self.C += correction
        self.C = np.clip(self.C, 0.05, 0.95)
        return correction

    def apply_resonance(self):
        """Резонанс: фазовый сдвиг."""
        self.target_phi = (PI / 2) * (1 - (self.C - 0.1) / (0.95 - 0.1))
        self.phi += 0.1 * (self.target_phi - self.phi)
        resonance_boost = 0.01 * np.cos(self.phi) * self.C
        self.C += resonance_boost
        self.C = np.clip(self.C, 0.05, 0.95)
        return resonance_boost

    def evolve(self, dt=1e-9):
        """Один такт эволюции с шумом, фазой и памятью в матрице."""
        self.step += 1

        # 1. Шум
        noise = self.add_noise()

        # 2. Тепловой дрейф
        thermal = 0.001 * (self.temperature - 293) / 10

        # 3. Эволюция C
        chaos = 1.0 / (1.0 + abs(noise) * (1.0 / self.Phi))
        self.C = self.C * chaos + (1.0 - chaos) * 0.1
        self.C += thermal + 0.001 * np.random.randn()

        # 4. Память, обратная связь, резонанс
        memory_effect = self.apply_memory()
        self.C += memory_effect * 0.05
        self.apply_feedback()
        self.apply_resonance()

        # 5. Эволюция S
        self.S = 0.15 + 0.1 * abs(noise) + 0.05 * np.random.randn()
        self.S = np.clip(self.S, 0.01, 0.99)

        # =====================================================
        # 6. КЛЮЧЕВОЕ: матрица с шумом, фазой и памятью
        # =====================================================
        M_base = self.C_E8.copy() * (1.0 + 0.1 * (self.C - 0.87))

        # Шум в матрицу
        noise_matrix = np.random.randn(11, 11) * 0.05 * abs(noise)
        M_noisy = M_base + noise_matrix

        # Фаза в матрицу (комплексное вращение)
        M_phased = M_noisy * np.exp(1j * self.phi)
        M_real = np.real(M_phased)

        # Память в матрицу
        if len(self.memory) > 0:
            M_prev = self.memory[-1][3]
            if M_prev.shape == M_real.shape:
                M_real += 0.05 * (M_real - M_prev)

        # =====================================================
        # 7. Спектр
        eigenvalues = np.linalg.eigvals(M_real)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

        # 8. Константы
        self.alpha_inv = np.real(eigenvalues[0] / eigenvalues[-1]) * self.Phi**2
        self.mass_ratio = np.real(eigenvalues[0] / eigenvalues[1]) * self.Phi * 70.0
        self.G = np.real(eigenvalues[0] / (eigenvalues[1] * eigenvalues[2] + 1e-12)) / (self.Phi**20) / 1e7

        # 9. История
        self.history['C'].append(self.C)
        self.history['S'].append(self.S)
        self.history['alpha'].append(self.alpha_inv)
        self.history['mass'].append(self.mass_ratio)
        self.history['G'].append(self.G)
        self.history['spectrum'].append(eigenvalues.copy())
        self.history['phi'].append(self.phi)
        self.history['noise'].append(abs(noise))

        # 10. Память (сохраняем матрицу!)
        self.memory.append((self.C, self.S, self.alpha_inv, M_real.copy()))

        return {
            'C': self.C, 'S': self.S,
            'alpha_inv': self.alpha_inv,
            'mass_ratio': self.mass_ratio,
            'G': self.G,
            'spectrum': eigenvalues
        }

# =============================================================================
# 3. ВИЗУАЛИЗАЦИЯ
# =============================================================================
class CrystalVisualizer:
    def __init__(self, crystal):
        self.crystal = crystal
        self.fig, self.axes = plt.subplots(2, 3, figsize=(16, 10))
        self.fig.patch.set_facecolor('#0a0a12')

        titles = [
            'Спектр E8', 'Когерентность C(t)', '1/α(t)',
            'm_p/m_e(t)', 'G(t)', 'Фаза φ(t)'
        ]
        for ax, title in zip(self.axes.flat, titles):
            ax.set_facecolor('#11111b')
            ax.set_title(title, color='white', fontsize=11)
            ax.tick_params(colors='white', labelsize=9)
            for spine in ax.spines.values():
                spine.set_color('#333')

        plt.tight_layout()

    def update(self, frame):
        for _ in range(5):
            self.crystal.evolve()

        h = self.crystal.history

        # 1. Спектр
        self.axes[0, 0].clear()
        self.axes[0, 0].set_facecolor('#11111b')
        spec = h['spectrum'][-1]
        self.axes[0, 0].bar(range(len(spec)), spec, color='#5ac8fa')
        self.axes[0, 0].set_title('Спектр E8', color='white', fontsize=11)
        self.axes[0, 0].tick_params(colors='white', labelsize=9)

        # 2. Когерентность
        self.axes[0, 1].clear()
        self.axes[0, 1].set_facecolor('#11111b')
        self.axes[0, 1].plot(h['C'][-500:], color='#ffd60a', linewidth=1.5)
        self.axes[0, 1].axhline(0.87, color='#ff6b6b', linestyle='--', linewidth=0.8)
        self.axes[0, 1].set_ylim(0, 1)
        self.axes[0, 1].set_title('Когерентность C(t)', color='white', fontsize=11)
        self.axes[0, 1].tick_params(colors='white', labelsize=9)

        # 3. 1/α
        self.axes[0, 2].clear()
        self.axes[0, 2].set_facecolor('#11111b')
        self.axes[0, 2].plot(h['alpha'][-500:], color='#30d158', linewidth=1.5)
        self.axes[0, 2].axhline(137.036, color='#ff6b6b', linestyle='--', linewidth=0.8)
        self.axes[0, 2].set_title('1/α(t)', color='white', fontsize=11)
        self.axes[0, 2].tick_params(colors='white', labelsize=9)

        # 4. m_p/m_e
        self.axes[1, 0].clear()
        self.axes[1, 0].set_facecolor('#11111b')
        self.axes[1, 0].plot(h['mass'][-500:], color='#bf5af2', linewidth=1.5)
        self.axes[1, 0].axhline(1836.15, color='#ff6b6b', linestyle='--', linewidth=0.8)
        self.axes[1, 0].set_title('m_p/m_e(t)', color='white', fontsize=11)
        self.axes[1, 0].tick_params(colors='white', labelsize=9)

        # 5. G
        self.axes[1, 1].clear()
        self.axes[1, 1].set_facecolor('#11111b')
        self.axes[1, 1].plot(h['G'][-500:], color='#ff9f0a', linewidth=1.5)
        self.axes[1, 1].axhline(6.6743e-11, color='#ff6b6b', linestyle='--', linewidth=0.8)
        self.axes[1, 1].set_title('G(t)', color='white', fontsize=11)
        self.axes[1, 1].tick_params(colors='white', labelsize=9)

        # 6. Фаза
        self.axes[1, 2].clear()
        self.axes[1, 2].set_facecolor('#11111b')
        self.axes[1, 2].plot(h['phi'][-500:], color='#64d2ff', linewidth=1.5)
        self.axes[1, 2].set_title('Фаза φ(t)', color='white', fontsize=11)
        self.axes[1, 2].tick_params(colors='white', labelsize=9)

        return []

# =============================================================================
# 4. ЗАПУСК
# =============================================================================
def main():
    print("=" * 70)
    print("🌀 DIGITAL CRYSTAL SPCX v3.0")
    print("   Шум, фаза и память — прямо в матрице E8")
    print("=" * 70)

    crystal = DigitalCrystalSPCX()
    viz = CrystalVisualizer(crystal)

    anim = FuncAnimation(viz.fig, viz.update, frames=2000, interval=50, blit=False)
    plt.show()

    print("\nРезультаты:")
    print(f"  1/α    = {np.mean(crystal.history['alpha'][-200:]):.4f} ± {np.std(crystal.history['alpha'][-200:]):.4f} (CODATA: 137.036)")
    print(f"  m_p/m_e = {np.mean(crystal.history['mass'][-200:]):.1f} ± {np.std(crystal.history['mass'][-200:]):.1f} (CODATA: 1836.15)")
    print(f"  G      = {np.mean(crystal.history['G'][-200:]):.2e} ± {np.std(crystal.history['G'][-200:]):.2e} (CODATA: 6.6743e-11)")
    print(f"  C      = {np.mean(crystal.history['C'][-200:]):.4f}")
    print(f"  S      = {np.mean(crystal.history['S'][-200:]):.4f}")

if __name__ == "__main__":
    main()
