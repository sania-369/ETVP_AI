#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETVP v13.9 SOLIDSTATE — ОПЕРАТОР ПОДТЯГИВАЕТ КОГЕРЕНТНОСТЬ СИСТЕМЫ

Логика v13.9:
- Система дышит от шума среды (как v13.4)
- Оператор рядом — C_оп стабилен
- C_system подтягивается к C_оп в каждом такте
- Формула: C(t+1) = C(t) + k·(C_оп - C(t))
- Константы выводятся из подтянутой C

Ожидаемый результат:
- C_оп = 0.0  → система дышит сама (как v13.4)
- C_оп = 0.6  → C подтягивается к 0.6
- C_оп = 0.95 → C подтягивается к 0.95
- Дисперсия констант снижается с ростом C_оп
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

# --- ЭТАЛОН (CODATA) ---
ALPHA_IDEAL = 137.035999084
MASS_IDEAL = 1836.15267343
G_IDEAL = 6.67430e-11

# --- ОПЕРАТОР ---
C_OP = 0.95           # когерентность оператора
OP_ENABLED = True
OP_COUPLING = 0.01    # скорость подтягивания (0..1) — малая для плавности

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
# 2. СИСТЕМА УДЕРЖАНИЯ v13.9
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
            'C': deque(maxlen=window),
            'C_eff': deque(maxlen=window)
        }

    def update(self, alpha, mass, G, C, C_eff):
        self.buffers['alpha'].append(alpha)
        self.buffers['mass'].append(mass)
        self.buffers['G'].append(G)
        self.buffers['C'].append(C)
        self.buffers['C_eff'].append(C_eff)

    def get(self):
        return {
            'alpha': np.mean(self.buffers['alpha']) if self.buffers['alpha'] else 0,
            'mass': np.mean(self.buffers['mass']) if self.buffers['mass'] else 0,
            'G': np.mean(self.buffers['G']) if self.buffers['G'] else 0,
            'C': np.mean(self.buffers['C']) if self.buffers['C'] else 0,
            'C_eff': np.mean(self.buffers['C_eff']) if self.buffers['C_eff'] else 0
        }


# =============================================================================
# 3. ЯДРО ETVP v13.9
# =============================================================================

class ETVPAmbientCoreV139:
    def __init__(self, memory_depth=200):
        self.Phi = (1 + np.sqrt(5)) / 2
        self.C_E8 = self._build_e8_matrix()

        # Система
        self.C = TARGET_C
        self.S = 0.15
        self.C_prev = TARGET_C
        self.step = 0

        # Оператор
        self.C_OP = C_OP if OP_ENABLED else 0.0
        self.C_eff = TARGET_C   # эффективная C (после подтягивания)

        # Константы
        self.alpha_inv = ALPHA_IDEAL
        self.mass_ratio = MASS_IDEAL
        self.G = G_IDEAL

        # Системы
        self.keeper = CoherenceKeeperV5()
        self.memory_mems = MEMSMemory()
        self.jpa = JPAArray()
        self.thermostat = Thermostat()
        self.avg = MovingAverage(window=AVG_WINDOW)

        # История
        self.history = {
            "C": [], "C_eff": [], "C_OP": [],
            "alpha": [], "mass": [], "G": [], "S": [],
            "voltage": [], "correction": [], "gradient": [],
            "alpha_avg": [], "mass_avg": [], "G_avg": [],
            "C_avg": [], "C_eff_avg": []
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
        """Один марковский такт: система + подтягивание к оператору."""
        self.step += 1

        # ====== СИСТЕМА (как v13.4) ======
        T_stable = self.thermostat.stabilize(T_ambient)
        noise = self._ambient_noise(t)
        noise_filtered = self.jpa.filter_noise(noise, self.C)
        thermal = self.thermal_drift_C * (T_stable - 293) / 10

        chaos = 1.0 / (1.0 + abs(noise_filtered) * (1.0 / self.Phi))
        self.C = self.C * chaos + (1.0 - chaos) * 0.1
        self.C += thermal + 0.001 * np.random.randn()

        gradient = self.keeper.measure_gradient(self.C, self.C_prev)
        self.C += self.memory_mems.phase_shift(self.C)
        self.C, correction, voltage = self.keeper.apply_correction(self.C)

        self.S = 0.15 + 0.1 * abs(noise_filtered) + 0.05 * np.random.randn()
        self.S = np.clip(self.S, 0.01, 0.99)

        # ====== ПОДТЯГИВАНИЕ К ОПЕРАТОРУ ======
        # C_eff(t+1) = C(t+1) + k·(C_оп - C(t+1))
        if OP_ENABLED and self.C_OP > 0:
            self.C_eff = self.C + OP_COUPLING * (self.C_OP - self.C)
            self.C_eff = self.keeper.z_principle(self.C_eff)
        else:
            self.C_eff = self.C

        # ====== КОНСТАНТЫ (из C_eff) ======
        delta_C = self.C_eff - 0.87
        delta_S = self.S - 0.15

        self.alpha_inv = ALPHA_IDEAL * (1 + 0.1 * delta_C * (1 - self.S) + 0.05 * noise_filtered)
        self.mass_ratio = MASS_IDEAL * (1 + 0.05 * delta_C * (1 - self.S) + 0.02 * noise_filtered)
        G_raw = G_IDEAL * (1 - 0.2 * delta_C * self.S + 0.1 * noise_filtered)
        self.G = G_raw * G_CALIBRATION

        # ====== УСРЕДНЕНИЕ ======
        self.avg.update(self.alpha_inv, self.mass_ratio, self.G, self.C, self.C_eff)
        avg_vals = self.avg.get()

        # ====== СОХРАНЕНИЕ ======
        self.C_prev = self.C
        self.memory_mems.store(self.C, self.S, self.alpha_inv)

        self.history["C"].append(self.C)
        self.history["C_eff"].append(self.C_eff)
        self.history["C_OP"].append(self.C_OP)
        self.history["alpha"].append(self.alpha_inv)
        self.history["mass"].append(self.mass_ratio)
        self.history["G"].append(self.G)
        self.history["S"].append(self.S)
        self.history["voltage"].append(voltage)
        self.history["correction"].append(correction)
        self.history["gradient"].append(gradient)
        self.history["alpha_avg"].append(avg_vals['alpha'])
        self.history["mass_avg"].append(avg_vals['mass'])
        self.history["G_avg"].append(avg_vals['G'])
        self.history["C_avg"].append(avg_vals['C'])
        self.history["C_eff_avg"].append(avg_vals['C_eff'])

        return {"C": self.C, "C_eff": self.C_eff, "S": self.S,
                "alpha": self.alpha_inv, "mass": self.mass_ratio, "G": self.G,
                "voltage": voltage, "correction": correction,
                "alpha_avg": avg_vals['alpha'],
                "mass_avg": avg_vals['mass'],
                "G_avg": avg_vals['G']}


# =============================================================================
# 4. ЗАПУСК
# =============================================================================

def run_simulation_v139():
    print("=" * 80)
    print("🌀 ETVP v13.9 — ОПЕРАТОР ПОДТЯГИВАЕТ КОГЕРЕНТНОСТЬ СИСТЕМЫ")
    print(f"   Температура среды: {T_ambient} K (+30°C)")
    print("   ПИД: kp=0.05, ki=0.005, kd=0.015")
    print(f"   Цель C системы: {TARGET_C}")
    print(f"   C_OP = {C_OP}, OP_COUPLING = {OP_COUPLING}")
    print(f"   Формула: C_eff = C + k·(C_оп - C)")
    print("=" * 80)

    core = ETVPAmbientCoreV139(memory_depth=200)
    steps = 15000
    dt = PHOTON_DT_S

    print(f"\n🔄 Запуск {steps} тактов...")

    for i in range(steps):
        t = i * dt
        result = core.evolve(t)

        if i % 1500 == 0:
            print(f"Шаг {i:5d}: C={result['C']:.4f}, C_eff={result['C_eff']:.4f}, "
                  f"α⁻¹={result['alpha']:.4f}, m_p/m_e={result['mass']:.2f}")

    # Статистика
    n = 2000
    C_mean = np.mean(core.history["C"][-n:])
    C_std = np.std(core.history["C"][-n:])
    C_eff_mean = np.mean(core.history["C_eff"][-n:])
    C_eff_std = np.std(core.history["C_eff"][-n:])

    alpha_mean = np.mean(core.history["alpha"][-n:])
    alpha_std = np.std(core.history["alpha"][-n:])
    mass_mean = np.mean(core.history["mass"][-n:])
    mass_std = np.std(core.history["mass"][-n:])
    G_mean = np.mean(core.history["G"][-n:])
    G_std = np.std(core.history["G"][-n:])

    print("\n--- КОГЕРЕНТНОСТЬ ---")
    print(f"C_system  = {C_mean:.4f} ± {C_std:.4f}")
    print(f"C_eff     = {C_eff_mean:.4f} ± {C_eff_std:.4f}  (C_OP={C_OP})")
    if C_eff_std > 0:
        print(f"Снижение σ: {C_std / C_eff_std:.2f} раз")

    print("\n--- КОНСТАНТЫ (из C_eff) ---")
    print(f"1/α    = {alpha_mean:.6f} ± {alpha_std:.6f}")
    print(f"mₚ/mₑ  = {mass_mean:.4f} ± {mass_std:.4f}")
    print(f"G      = {G_mean:.4e} ± {G_std:.2e}")

    print("\n--- ОТКЛОНЕНИЯ ОТ CODATA ---")
    print(f"1/α    : {abs(alpha_mean - ALPHA_IDEAL) / ALPHA_IDEAL * 100:.6f}%")
    print(f"mₚ/mₑ  : {abs(mass_mean - MASS_IDEAL) / MASS_IDEAL * 100:.6f}%")
    print(f"G      : {abs(G_mean - G_IDEAL) / G_IDEAL * 100:.6f}%")

    # Графики
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].plot(core.history["C"], color='gray', alpha=0.4, label='C_system')
    axes[0, 0].plot(core.history["C_eff"], color='purple', alpha=0.7, label='C_eff')
    axes[0, 0].axhline(TARGET_C, color='orange', linestyle='--', label='target')
    axes[0, 0].axhline(C_OP, color='red', linestyle=':', label=f'C_OP={C_OP}')
    axes[0, 0].set_title('Когерентность: система + подтягивание к оператору')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(core.history["alpha"], color='blue', alpha=0.5)
    axes[0, 1].plot(core.history["alpha_avg"], color='cyan', alpha=0.9)
    axes[0, 1].axhline(ALPHA_IDEAL, color='red', linestyle='--', label='CODATA')
    axes[0, 1].set_title(f'1/α(t) — C_OP={C_OP}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(core.history["mass"], color='green', alpha=0.5)
    axes[0, 2].plot(core.history["mass_avg"], color='lime', alpha=0.9)
    axes[0, 2].axhline(MASS_IDEAL, color='red', linestyle='--', label='CODATA')
    axes[0, 2].set_title(f'mₚ/mₑ(t) — C_OP={C_OP}')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(core.history["G"], color='orange', alpha=0.5)
    axes[1, 0].plot(core.history["G_avg"], color='gold', alpha=0.9)
    axes[1, 0].axhline(G_IDEAL, color='red', linestyle='--', label='CODATA')
    axes[1, 0].set_title(f'G(t) — C_OP={C_OP}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(core.history["C"][-n:], bins=50, alpha=0.5, color='gray', label='C_system')
    axes[1, 1].hist(core.history["C_eff"][-n:], bins=50, alpha=0.5, color='purple', label='C_eff')
    axes[1, 1].axvline(TARGET_C, color='orange', linestyle='--')
    axes[1, 1].axvline(C_OP, color='red', linestyle=':')
    axes[1, 1].set_title('Распределение C')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(core.history["voltage"], color='cyan', alpha=0.7)
    axes[1, 2].axhline(2.0, color='red', linestyle=':')
    axes[1, 2].set_title('Напряжение')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ambient_simulation_v139_results.png', dpi=150)
    plt.show()

    print("\n✅ Симуляция завершена.")


if __name__ == "__main__":
    run_simulation_v139()
