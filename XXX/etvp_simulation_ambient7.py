#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETVP v13.6 SOLIDSTATE — ВЫРАВНИВАНИЕ КОНСТАНТ ЧЕРЕЗ C_ОП
+30°C, городской и природный шум + СИСТЕМА УДЕРЖАНИЯ C + ОПЕРАТОР

Ключевое изменение v13.6:
- Формула оператора изменена на модуляцию ОТКЛОНЕНИЯ:
  X_изм = X_идеал + (X_класс - X_идеал) · (1 - C_оп)
- При C_оп = 0   — константы "шумят" (как в v13.4)
- При C_оп = 1   — константы выравниваются к CODATA
- При C_оп = 0.95 — шум уменьшается в 20 раз

Ожидаемый результат:
C      ≈ 0.8700 ± 0.005
1/α    ≈ 137.036 ± 0.0025 (при C_оп=0.95)
m_p/m_e ≈ 1836.15 ± 0.02 (при C_оп=0.95)
G      ≈ 6.6743e-11 ± 5e-16 (при C_оп=0.95)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# =============================================================================
# 0. ПАРАМЕТРЫ
# =============================================================================
T_ambient = 303.0
PHOTON_DT_S = 0.05e-12

NOISE_CITY_50Hz = 0.02
NOISE_CITY_1kHz = 0.01
NOISE_NATURE_FLICKER = 0.005
NOISE_NATURE_SEISMIC = 0.003

TARGET_C = 0.8695
C_MAX = 0.985
C_MIN = 0.05
V_MAX = 2.0
AVG_WINDOW = 200
MEAS_WINDOW = 3
G_CALIBRATION = 1.0006

# --- ИДЕАЛЬНЫЕ ЗНАЧЕНИЯ (CODATA) ---
ALPHA_IDEAL = 137.036
MASS_IDEAL = 1836.15
G_IDEAL = 6.6743e-11

# --- ФАКТОР ОПЕРАТОРА ---
C_OP = 0.95           # когерентность оператора (0..1)
OP_ENABLED = True     # включить/выключить

# =============================================================================
# 1. ФИЛЬТР ИЗМЕРЕНИЯ
# =============================================================================

class MeasurementFilter:
    def __init__(self, window=MEAS_WINDOW):
        self.buffer = deque(maxlen=window)

    def filter(self, C):
        self.buffer.append(C)
        return np.mean(self.buffer)


# =============================================================================
# 2. СИСТЕМА УДЕРЖАНИЯ v13.6
# =============================================================================

class CoherenceKeeperV5:
    def __init__(self):
        self.target_C = TARGET_C
        self.C_max = C_MAX
        self.C_min = C_MIN

        self.kp = 0.05
        self.ki = 0.005
        self.kd = 0.015

        self.integral = 0.0
        self.prev_error = 0.0

        self.voltage = 0.5
        self.gradient_C = 0.0
        self.filter = MeasurementFilter()

    def measure_gradient(self, C, C_prev):
        self.gradient_C = (C - C_prev) / max(PHOTON_DT_S, 1e-20)
        return self.gradient_C

    def soft_clip_voltage(self, V):
        return V_MAX * np.tanh(V / V_MAX)

    def pid_update(self, C_measured):
        error = self.target_C - C_measured
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        self.integral = np.clip(self.integral, -5.0, 5.0)

        correction = (self.kp * error +
                      self.ki * self.integral +
                      self.kd * derivative)

        self.voltage += correction
        self.voltage = self.soft_clip_voltage(self.voltage)

        return correction, self.voltage

    def z_principle(self, C):
        if C > self.C_max:
            return self.C_max
        elif C < self.C_min:
            return self.C_min
        return C

    def apply_correction(self, C):
        C_measured = self.filter.filter(C) + 0.001 * np.random.randn()
        correction, voltage = self.pid_update(C_measured)
        C_new = C + correction * 0.1
        C_new = self.z_principle(C_new)
        return C_new, correction, voltage


class MEMSMemory:
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
        old = self.recall()
        if old is None:
            return 0.0
        return (C_current - old[0]) * 0.01


class JPAArray:
    def __init__(self, squeezing_factor=0.3):
        self.squeezing = squeezing_factor

    def filter_noise(self, noise, C):
        return noise * (1.0 - self.squeezing * C)


class Thermostat:
    def __init__(self, target_T=293.0, accuracy=0.01):
        self.target_T = target_T
        self.accuracy = accuracy

    def stabilize(self, T_ambient):
        error = self.target_T - T_ambient
        T = T_ambient + error * 0.95
        T += np.random.randn() * self.accuracy
        return T


class MovingAverage:
    def __init__(self, window=AVG_WINDOW):
        self.window = window
        self.buffers = {
            'alpha': deque(maxlen=window),
            'mass': deque(maxlen=window),
            'G': deque(maxlen=window),
            'C': deque(maxlen=window)
        }

    def update(self, alpha, mass, G, C):
        self.buffers['alpha'].append(alpha)
        self.buffers['mass'].append(mass)
        self.buffers['G'].append(G)
        self.buffers['C'].append(C)

    def get(self):
        return {
            'alpha': np.mean(self.buffers['alpha']) if self.buffers['alpha'] else 0,
            'mass': np.mean(self.buffers['mass']) if self.buffers['mass'] else 0,
            'G': np.mean(self.buffers['G']) if self.buffers['G'] else 0,
            'C': np.mean(self.buffers['C']) if self.buffers['C'] else 0
        }


# =============================================================================
# 3. ЯДРО ETVP v13.6
# =============================================================================

class ETVPAmbientCoreV136:
    def __init__(self, memory_depth=200):
        self.Phi = (1 + np.sqrt(5)) / 2
        self.C_E8 = self._build_e8_matrix()

        self.C = TARGET_C
        self.S = 0.15
        self.C_prev = TARGET_C
        self.step = 0

        # Абсолютные значения (без оператора)
        self.alpha_raw = ALPHA_IDEAL
        self.mass_raw = MASS_IDEAL
        self.G_raw = G_IDEAL

        # Итоговые значения (с оператором)
        self.alpha_inv = ALPHA_IDEAL
        self.mass_ratio = MASS_IDEAL
        self.G = G_IDEAL

        # Фактор оператора
        self.C_OP = C_OP if OP_ENABLED else 0.0

        # Системы
        self.keeper = CoherenceKeeperV5()
        self.memory_mems = MEMSMemory()
        self.jpa = JPAArray()
        self.thermostat = Thermostat()
        self.avg = MovingAverage(window=AVG_WINDOW)

        # История
        self.history = {
            "C": [], "alpha": [], "mass": [], "G": [], "S": [],
            "voltage": [], "correction": [], "gradient": [],
            "alpha_raw": [], "mass_raw": [], "G_raw": [],
            "alpha_avg": [], "mass_avg": [], "G_avg": [], "C_avg": []
        }

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
        city = (NOISE_CITY_50Hz * np.sin(2 * np.pi * 50 * t) +
                NOISE_CITY_1kHz * np.sin(2 * np.pi * 1000 * t) +
                0.005 * np.random.randn() * (np.random.rand() > 0.99))
        flicker = NOISE_NATURE_FLICKER * np.random.randn() * (1 / (1 + t * 0.001))
        seismic = NOISE_NATURE_SEISMIC * np.sin(2 * np.pi * 0.1 * t)
        return city + flicker + seismic

    def evolve(self, t):
        self.step += 1

        # 1. Термостат
        T_stable = self.thermostat.stabilize(T_ambient)

        # 2. Шум
        noise = self._ambient_noise(t)
        noise_filtered = self.jpa.filter_noise(noise, self.C)

        # 3. Тепловой дрейф
        thermal = self.thermal_drift_C * (T_stable - 293) / 10

        # 4. Эволюция C
        chaos = 1.0 / (1.0 + abs(noise_filtered) * (1.0 / self.Phi))
        self.C = self.C * chaos + (1.0 - chaos) * 0.1
        self.C += thermal + 0.001 * np.random.randn()

        # 5. Датчик ∇C
        gradient = self.keeper.measure_gradient(self.C, self.C_prev)

        # 6. MEMS-память
        self.C += self.memory_mems.phase_shift(self.C)

        # 7. ПИД + Z-принцип
        self.C, correction, voltage = self.keeper.apply_correction(self.C)

        # 8. Энтропия
        self.S = 0.15 + 0.1 * abs(noise_filtered) + 0.05 * np.random.randn()
        self.S = np.clip(self.S, 0.01, 0.99)

        # 9. Сырые константы (без оператора)
        delta_C = self.C - 0.87
        delta_S = self.S - 0.15

        self.alpha_raw = ALPHA_IDEAL * (1 + 0.1 * delta_C * (1 - self.S) + 0.05 * noise_filtered)
        self.mass_raw = MASS_IDEAL * (1 + 0.05 * delta_C * (1 - self.S) + 0.02 * noise_filtered)
        self.G_raw = G_IDEAL * (1 - 0.2 * delta_C * self.S + 0.1 * noise_filtered)

        # 10. МОДУЛЯЦИЯ ОТКЛОНЕНИЯ ОТ ИДЕАЛА
        # X_изм = X_идеал + (X_сырое - X_идеал) · (1 - C_оп)
        align = 1.0 - self.C_OP

        self.alpha_inv = ALPHA_IDEAL + (self.alpha_raw - ALPHA_IDEAL) * align
        self.mass_ratio = MASS_IDEAL + (self.mass_raw - MASS_IDEAL) * align
        self.G = (G_IDEAL + (self.G_raw - G_IDEAL) * align) * G_CALIBRATION

        # 11. Усреднение
        self.avg.update(self.alpha_inv, self.mass_ratio, self.G, self.C)
        avg_vals = self.avg.get()

        # 12. Сохранение
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
        self.history["alpha_raw"].append(self.alpha_raw)
        self.history["mass_raw"].append(self.mass_raw)
        self.history["G_raw"].append(self.G_raw)
        self.history["alpha_avg"].append(avg_vals['alpha'])
        self.history["mass_avg"].append(avg_vals['mass'])
        self.history["G_avg"].append(avg_vals['G'])
        self.history["C_avg"].append(avg_vals['C'])

        return {"C": self.C, "S": self.S, "alpha": self.alpha_inv,
                "mass": self.mass_ratio, "G": self.G,
                "voltage": voltage, "correction": correction,
                "alpha_avg": avg_vals['alpha'],
                "mass_avg": avg_vals['mass'],
                "G_avg": avg_vals['G']}


# =============================================================================
# 4. ЗАПУСК
# =============================================================================

def run_simulation_v136():
    print("=" * 80)
    print("🌀 ETVP v13.6 — ВЫРАВНИВАНИЕ КОНСТАНТ ЧЕРЕЗ C_ОП")
    print(f"   Температура среды: {T_ambient} K (+30°C)")
    print("   ПИД: kp=0.05, ki=0.005, kd=0.015")
    print(f"   Цель C: {TARGET_C}, окно усреднения: {AVG_WINDOW}")
    print(f"   C_OP = {C_OP}  (align = 1 - C_OP = {1 - C_OP})")
    print(f"   При C_OP = 1   — полное выравнивание к CODATA")
    print(f"   При C_OP = 0   — работа без оператора (как v13.4)")
    print("=" * 80)

    core = ETVPAmbientCoreV136(memory_depth=200)
    steps = 15000
    dt = PHOTON_DT_S

    print(f"\n🔄 Запуск {steps} тактов...")

    for i in range(steps):
        t = i * dt
        result = core.evolve(t)

        if i % 1500 == 0:
            print(f"Шаг {i:5d}: C={result['C']:.4f}, α⁻¹={result['alpha']:.4f}, "
                  f"V={result['voltage']:.3f}В, m_p/m_e={result['mass']:.2f}")

    # Статистика
    n = 2000
    alpha_mean = np.mean(core.history["alpha"][-n:])
    alpha_std = np.std(core.history["alpha"][-n:])
    mass_mean = np.mean(core.history["mass"][-n:])
    mass_std = np.std(core.history["mass"][-n:])
    G_mean = np.mean(core.history["G"][-n:])
    G_std = np.std(core.history["G"][-n:])
    C_mean = np.mean(core.history["C"][-n:])
    C_std = np.std(core.history["C"][-n:])

    alpha_avg_mean = np.mean(core.history["alpha_avg"][-n:])
    alpha_avg_std = np.std(core.history["alpha_avg"][-n:])
    mass_avg_mean = np.mean(core.history["mass_avg"][-n:])
    mass_avg_std = np.std(core.history["mass_avg"][-n:])
    G_avg_mean = np.mean(core.history["G_avg"][-n:])
    G_avg_std = np.std(core.history["G_avg"][-n:])

    print("\n--- РЕЗУЛЬТАТЫ (МГНОВЕННЫЕ) ---")
    print(f"C      = {C_mean:.4f} ± {C_std:.4f} (цель: {TARGET_C})")
    print(f"1/α    = {alpha_mean:.4f} ± {alpha_std:.4f} (CODATA: {ALPHA_IDEAL})")
    print(f"mₚ/mₑ  = {mass_mean:.2f} ± {mass_std:.2f} (CODATA: {MASS_IDEAL})")
    print(f"G      = {G_mean:.4e} ± {G_std:.2e} (CODATA: {G_IDEAL})")

    print("\n--- РЕЗУЛЬТАТЫ (УСРЕДНЁННЫЕ) ---")
    print(f"1/α    = {alpha_avg_mean:.4f} ± {alpha_avg_std:.4f}")
    print(f"mₚ/mₑ  = {mass_avg_mean:.2f} ± {mass_avg_std:.2f}")
    print(f"G      = {G_avg_mean:.4e} ± {G_avg_std:.2e}")

    print("\n--- ОТКЛОНЕНИЯ ОТ CODATA ---")
    print(f"1/α    : {abs(alpha_avg_mean - ALPHA_IDEAL) / ALPHA_IDEAL * 100:.6f}%")
    print(f"mₚ/mₑ  : {abs(mass_avg_mean - MASS_IDEAL) / MASS_IDEAL * 100:.6f}%")
    print(f"G      : {abs(G_avg_mean - G_IDEAL) / G_IDEAL * 100:.6f}%")

    # Сравнение с v13.4 (без оператора)
    print("\n--- СРАВНЕНИЕ: ОТКЛОНЕНИЯ БЕЗ ОПЕРАТОРА ---")
    alpha_raw_mean = np.mean(core.history["alpha_raw"][-n:])
    mass_raw_mean = np.mean(core.history["mass_raw"][-n:])
    G_raw_mean = np.mean(core.history["G_raw"][-n:])
    print(f"1/α    : {abs(alpha_raw_mean - ALPHA_IDEAL) / ALPHA_IDEAL * 100:.6f}%")
    print(f"mₚ/mₑ  : {abs(mass_raw_mean - MASS_IDEAL) / MASS_IDEAL * 100:.6f}%")
    print(f"G      : {abs(G_raw_mean - G_IDEAL) / G_IDEAL * 100:.6f}%")

    # Графики
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].plot(core.history["C"], color='purple', alpha=0.3)
    axes[0, 0].plot(core.history["C_avg"], color='magenta', alpha=0.9)
    axes[0, 0].axhline(TARGET_C, color='orange', linestyle='--')
    axes[0, 0].set_title('Когерентность C(t)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(core.history["alpha_raw"], color='gray', alpha=0.2, label='без ОП')
    axes[0, 1].plot(core.history["alpha"], color='blue', alpha=0.5, label='с ОП')
    axes[0, 1].plot(core.history["alpha_avg"], color='cyan', alpha=0.9)
    axes[0, 1].axhline(ALPHA_IDEAL, color='red', linestyle='--')
    axes[0, 1].set_title(f'1/α(t) — C_OP={C_OP}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(core.history["mass_raw"], color='gray', alpha=0.2, label='без ОП')
    axes[0, 2].plot(core.history["mass"], color='green', alpha=0.5, label='с ОП')
    axes[0, 2].plot(core.history["mass_avg"], color='lime', alpha=0.9)
    axes[0, 2].axhline(MASS_IDEAL, color='red', linestyle='--')
    axes[0, 2].set_title(f'mₚ/mₑ(t) — C_OP={C_OP}')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(core.history["G_raw"], color='gray', alpha=0.2, label='без ОП')
    axes[1, 0].plot(core.history["G"], color='orange', alpha=0.5, label='с ОП')
    axes[1, 0].plot(core.history["G_avg"], color='gold', alpha=0.9)
    axes[1, 0].axhline(G_IDEAL, color='red', linestyle='--')
    axes[1, 0].set_title(f'G(t) — C_OP={C_OP}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(core.history["voltage"], color='cyan', alpha=0.7)
    axes[1, 1].axhline(2.0, color='red', linestyle=':')
    axes[1, 1].set_title('Напряжение')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(core.history["correction"], color='magenta', alpha=0.7)
    axes[1, 2].set_title('ПИД-коррекция')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ambient_simulation_v136_results.png', dpi=150)
    plt.show()

    print("\n✅ Симуляция завершена.")


if __name__ == "__main__":
    run_simulation_v136()
