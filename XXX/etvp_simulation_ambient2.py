#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETVP v13.0 SOLIDSTATE — СИМУЛЯЦИЯ С УДЕРЖАНИЕМ КОГЕРЕНТНОСТИ
+30°C, городской и природный шум + СИСТЕМА УДЕРЖАНИЯ C

Добавлено:
- Датчик ∇C (измерение когерентности)
- pn-переходы (модуляция C через напряжение)
- JPA-массив (адаптивный шум)
- MEMS-память (рекурсивная задержка)
- ПИД-контроллер (удержание целевой C)
- Z-принцип (клиппинг на 0.985)
- Термостат (стабилизация температуры)

Результат: стабильные константы, устойчивая C, минимальный дрейф.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# =============================================================================
# 0. ПАРАМЕТРЫ СРЕДЫ
# =============================================================================
T_ambient = 303.0  # +30°C
PHOTON_DT_S = 0.05e-12

# Шумовые параметры
NOISE_CITY_50Hz = 0.02
NOISE_CITY_1kHz = 0.01
NOISE_NATURE_FLICKER = 0.005
NOISE_NATURE_SEISMIC = 0.003

# Параметры удержания
TARGET_C = 0.87
C_MAX = 0.985
C_MIN = 0.05
C_CRITICAL = 0.92  # для Z-принципа

# =============================================================================
# 1. СИСТЕМА УДЕРЖАНИЯ КОГЕРЕНТНОСТИ
# =============================================================================

class CoherenceKeeper:
    """Система удержания когерентности с ПИД-контроллером."""

    def __init__(self):
        self.target_C = TARGET_C
        self.C_max = C_MAX
        self.C_min = C_MIN

        # ПИД-коэффициенты
        self.kp = 0.05
        self.ki = 0.001
        self.kd = 0.01
        self.integral = 0.0
        self.prev_error = 0.0

        # Состояние
        self.voltage = 0.5  # напряжение на pn-переходах
        self.gradient_C = 0.0  # измеренный ∇C

    def measure_gradient(self, C, C_prev):
        """Датчик ∇C: измеряет разность когерентностей."""
        self.gradient_C = (C - C_prev) / max(PHOTON_DT_S, 1e-20)
        return self.gradient_C

    def pid_update(self, C_measured):
        """ПИД-коррекция на основе измерения C."""
        error = self.target_C - C_measured
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        correction = (self.kp * error +
                      self.ki * self.integral +
                      self.kd * derivative)

        # Напряжение на pn-переходах
        self.voltage += correction
        self.voltage = np.clip(self.voltage, 0.0, 2.0)

        return correction, self.voltage

    def z_principle(self, C):
        """Z-принцип: клиппинг когерентности."""
        if C > self.C_max:
            return self.C_max
        elif C < self.C_min:
            return self.C_min
        return C

    def apply_correction(self, C, C_measured):
        """Применяет коррекцию через pn-переходы."""
        correction, voltage = self.pid_update(C_measured)
        # pn-переходы модулируют C через напряжение
        C_new = C + correction * 0.1
        C_new = self.z_principle(C_new)
        return C_new, correction, voltage


class MEMSMemory:
    """MEMS-память: задержка сигнала через спиральный шлейф."""

    def __init__(self, depth=50, delay_steps=10):
        self.buffer = deque(maxlen=depth)
        self.delay_steps = delay_steps

    def store(self, C, S, alpha):
        self.buffer.append((C, S, alpha))

    def recall(self):
        if len(self.buffer) < self.delay_steps:
            return None
        return self.buffer[-self.delay_steps]

    def phase_shift(self, C_current):
        """Возвращает фазовый сдвиг от памяти."""
        old = self.recall()
        if old is None:
            return 0.0
        C_old = old[0]
        return (C_current - C_old) * 0.05


class JPAArray:
    """JPA-массив: адаптивное подавление шума."""

    def __init__(self, squeezing_factor=0.3):
        self.squeezing = squeezing_factor

    def adapt(self, C):
        """Чем выше C, тем сильнее сжатие шума."""
        return self.squeezing * (1.0 - C)

    def filter_noise(self, noise, C):
        """Подавляет шум пропорционально C."""
        return noise * (1.0 - self.squeezing * C)


class Thermostat:
    """Термостат: стабилизация температуры."""

    def __init__(self, target_T=293.0, accuracy=0.01):
        self.target_T = target_T
        self.accuracy = accuracy
        self.current_T = target_T

    def stabilize(self, T_ambient):
        """Стабилизирует температуру с заданной точностью."""
        error = self.target_T - T_ambient
        self.current_T = T_ambient + error * 0.9
        self.current_T += np.random.randn() * self.accuracy
        return self.current_T


# =============================================================================
# 2. ЯДРО ETVP v13.0 С УДЕРЖАНИЕМ
# =============================================================================

class ETVPAmbientCoreV13:
    def __init__(self, memory_depth=200):
        self.Phi = (1 + np.sqrt(5)) / 2
        self.C_E8 = self._build_e8_matrix()

        # Состояние
        self.C = TARGET_C
        self.S = 0.15
        self.C_prev = TARGET_C
        self.step = 0
        self.alpha_inv = 137.036
        self.mass_ratio = 1836.15
        self.G = 6.6743e-11

        # Система удержания
        self.keeper = CoherenceKeeper()
        self.memory_mems = MEMSMemory()
        self.jpa = JPAArray()
        self.thermostat = Thermostat()

        # История
        self.history = {
            "C": [], "alpha": [], "mass": [], "G": [], "S": [],
            "voltage": [], "correction": [], "gradient": [], "T": []
        }

        # Тепловой дрейф
        self.thermal_drift_C = 0.001 * (T_ambient - 293) / 10

    def _build_e8_matrix(self):
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

    def _ambient_noise(self, t):
        """Естественный шум среды."""
        city = (NOISE_CITY_50Hz * np.sin(2 * np.pi * 50 * t) +
                NOISE_CITY_1kHz * np.sin(2 * np.pi * 1000 * t) +
                0.005 * np.random.randn() * (np.random.rand() > 0.99))
        flicker = NOISE_NATURE_FLICKER * np.random.randn() * (1 / (1 + t * 0.001))
        seismic = NOISE_NATURE_SEISMIC * np.sin(2 * np.pi * 0.1 * t)
        return city + flicker + seismic

    def evolve(self, t):
        """Один такт эволюции с удержанием."""
        self.step += 1

        # 1. Термостат
        T_stable = self.thermostat.stabilize(T_ambient)

        # 2. Шум среды
        noise = self._ambient_noise(t)

        # 3. JPA-фильтрация шума
        noise_filtered = self.jpa.filter_noise(noise, self.C)

        # 4. Тепловой дрейф (с учётом термостата)
        thermal = self.thermal_drift_C * (T_stable - 293) / 10

        # 5. Эволюция C (естественное падение)
        chaos = 1.0 / (1.0 + abs(noise_filtered) * (1.0 / self.Phi))
        self.C = self.C * chaos + (1.0 - chaos) * 0.1
        self.C += thermal + 0.001 * np.random.randn()

        # 6. Датчик ∇C
        gradient = self.keeper.measure_gradient(self.C, self.C_prev)

        # 7. MEMS-память
        memory_shift = self.memory_mems.phase_shift(self.C)
        self.C += memory_shift

        # 8. ПИД-коррекция + Z-принцип
        C_measured = self.C + 0.005 * np.random.randn()  # шум измерения
        self.C, correction, voltage = self.keeper.apply_correction(self.C, C_measured)

        # 9. Энтропия
        self.S = 0.15 + 0.1 * abs(noise_filtered) + 0.05 * np.random.randn()
        self.S = np.clip(self.S, 0.01, 0.99)

        # 10. Константы
        delta_C = self.C - 0.87
        delta_S = self.S - 0.15

        self.alpha_inv = 137.036 * (1 + 0.1 * delta_C * (1 - self.S) + 0.05 * noise_filtered)
        self.mass_ratio = 1836.15 * (1 + 0.05 * delta_C * (1 - self.S) + 0.02 * noise_filtered)
        self.G = 6.6743e-11 * (1 - 0.2 * delta_C * self.S + 0.1 * noise_filtered)

        # 11. Сохранение
        self.C_prev = self.C
        self.memory_mems.store(self.C, self.S, self.alpha_inv)

        self.history["C"].append(self.C)
        self.history["alpha"].append(self.alpha_inv)
        self.history["mass"].append(self.mass_ratio)
        self.history["G"].append(self.G)
        self.history["S"].append(self.S)
        self.history["voltage"].append(voltage)
        self.history["correction"].append(correction)
        self.history["gradient"].append(gradient)
        self.history["T"].append(T_stable)

        return {"C": self.C, "S": self.S, "alpha": self.alpha_inv,
                "mass": self.mass_ratio, "G": self.G,
                "voltage": voltage, "correction": correction}


# =============================================================================
# 3. ЗАПУСК
# =============================================================================

def run_simulation_v13():
    print("=" * 80)
    print("🌀 ETVP v13.0 — СИМУЛЯЦИЯ С УДЕРЖАНИЕМ КОГЕРЕНТНОСТИ")
    print(f"   Температура среды: {T_ambient} K (+30°C)")
    print("   Устройства: датчик ∇C, pn-переходы, JPA, MEMS, ПИД, Z-принцип, термостат")
    print("=" * 80)

    core = ETVPAmbientCoreV13(memory_depth=200)
    steps = 10000
    dt = PHOTON_DT_S

    print(f"\n🔄 Запуск {steps} тактов...")

    for i in range(steps):
        t = i * dt
        result = core.evolve(t)

        if i % 1000 == 0:
            print(f"Шаг {i:5d}: C={result['C']:.4f}, α⁻¹={result['alpha']:.3f}, "
                  f"V={result['voltage']:.3f}В, m_p/m_e={result['mass']:.1f}")

    # Статистика
    alpha_mean = np.mean(core.history["alpha"][-1000:])
    alpha_std = np.std(core.history["alpha"][-1000:])
    mass_mean = np.mean(core.history["mass"][-1000:])
    mass_std = np.std(core.history["mass"][-1000:])
    G_mean = np.mean(core.history["G"][-1000:])
    G_std = np.std(core.history["G"][-1000:])
    C_mean = np.mean(core.history["C"][-1000:])
    C_std = np.std(core.history["C"][-1000:])

    print("\n--- РЕЗУЛЬТАТЫ С УДЕРЖАНИЕМ ---")
    print(f"C      = {C_mean:.4f} ± {C_std:.4f} (цель: {TARGET_C})")
    print(f"1/α    = {alpha_mean:.4f} ± {alpha_std:.4f} (CODATA: 137.036)")
    print(f"mₚ/mₑ  = {mass_mean:.1f} ± {mass_std:.1f} (CODATA: 1836.15)")
    print(f"G      = {G_mean:.2e} ± {G_std:.2e} (CODATA: 6.6743e-11)")

    # Графики (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].plot(core.history["C"], color='purple', alpha=0.7)
    axes[0, 0].axhline(TARGET_C, color='orange', linestyle='--', label='Target C')
    axes[0, 0].axhline(C_MAX, color='red', linestyle=':', label='Z-клиппинг')
    axes[0, 0].set_title(f'Когерентность C(t) с удержанием')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(core.history["alpha"], color='blue', alpha=0.7)
    axes[0, 1].axhline(137.036, color='red', linestyle='--', label='CODATA')
    axes[0, 1].set_title('1/α(t)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(core.history["mass"], color='green', alpha=0.7)
    axes[0, 2].axhline(1836.15, color='red', linestyle='--', label='CODATA')
    axes[0, 2].set_title('mₚ/mₑ(t)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(core.history["G"], color='orange', alpha=0.7)
    axes[1, 0].axhline(6.6743e-11, color='red', linestyle='--', label='CODATA')
    axes[1, 0].set_title('G(t)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(core.history["voltage"], color='cyan', alpha=0.7)
    axes[1, 1].set_title('Напряжение на pn-переходах')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(core.history["correction"], color='magenta', alpha=0.7)
    axes[1, 2].set_title('ПИД-коррекция')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ambient_simulation_v13_results.png', dpi=150)
    plt.show()

    print("\n✅ Симуляция завершена. Графики сохранены.")


if __name__ == "__main__":
    run_simulation_v13()
