import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =============================================================================
# 🌀 DYNAMIC FLOW: ОТО + КТП с дыханием Ψ-поля (на базе ETVP v12.3)
# =============================================================================

class LivingPhysics:
    """
    Сшивка ОТО и КТП с живой динамикой.
    Вместо статичных констант — дышащие параметры порядка C(t).
    """
    def __init__(self):
        # Геометрический базис ЕТВП
        self.Phi = (1.0 + np.sqrt(5.0)) / 2.0
        self.C_min = 1.0 / (self.Phi ** 10)
        self.C_max = 1.0 - 1.0 / (self.Phi ** 20)
        self.C = 1.0 - 1.0 / (self.Phi ** 12)  # Целевая когерентность
        
        # Физические константы (начальные, они будут «дышать»)
        self.G0 = 6.67430e-11  # Гравитационная постоянная
        self.alpha_em = 1.0 / 137.035999084  # Постоянная тонкой структуры
        
        # История для графиков
        self.history = {"C": [], "G": [], "alpha": [], "a": [], "H": []}
    
    def _tanh_limit(self, value, c_min, c_max):
        """Z-принцип: нелинейное удержание (из ETVP v12.3)"""
        epsilon = 1e-12
        E = (value - c_min) / (c_max - c_min + epsilon)
        E_limited = np.tanh(E) * 0.5 + 0.5
        return c_min + E_limited * (c_max - c_min)
    
    def evolve_coherence(self, entropy_flux):
        """Эволюция параметра порядка C(t)"""
        chaos_operator = 1.0 / (1.0 + abs(entropy_flux) * (1.0 / self.Phi))
        self.C = self.C * chaos_operator + (1.0 - chaos_operator) * self.C_min
        self.C = self._tanh_limit(self.C, self.C_min, self.C_max)
        return self.C
    
    # --- Сектор ОТО (Гравитация) ---
    
    def living_G(self):
        """
        Гравитационная постоянная, зависящая от когерентности.
        G(C) = G0 * (C_max - C) / (C_max - C_min)  — гравитация усиливается при падении C.
        Это отражает идею ЕТВП: хаос «утяжеляет» пространство.
        """
        return self.G0 * (self.C_max - self.C) / (self.C_max - self.C_min + 1e-12)
    
    def friedmann_equations(self, t, y):
        """
        Уравнения Фридмана с живой G(C) и живой тёмной энергией.
        y[0] = a (масштабный фактор)
        y[1] = a_dot
        """
        a, a_dot = y[0], y[1]
        G = self.living_G()
        
        # Плотность материи (упрощённо: пыль ~ 1/a^3)
        rho_m = 1.0 / (a**3 + 1e-12)
        
        # Плотность тёмной энергии (из дыхания поля)
        rho_lambda = 0.7 * (1.0 - self.C)  # Чем ниже C, тем сильнее тёмная энергия
        
        # Уравнение Фридмана: (a_dot / a)^2 = (8πG/3) * (ρ_m + ρ_Λ)
        H_sq = (8 * np.pi * G / 3.0) * (rho_m + rho_lambda)
        H = np.sqrt(max(H_sq, 0))
        
        # Ускорение
        a_ddot = - (4 * np.pi * G / 3.0) * (rho_m + 3 * (-rho_lambda)) * a
        
        return [a_dot, a_ddot]
    
    # --- Сектор КТП (Калибровочные поля) ---
    
    def living_alpha(self):
        """
        Бегущая константа связи α(C).
        В ЕТВП α⁻¹ ~ λ₀/λ₁. При C → C_max, α → α_em.
        """
        # Имитация бета-функции с дыханием
        beta = (1.0 - self.C) * 0.1  # Чем ниже C, тем сильнее бег
        alpha_inv = 1.0 / self.alpha_em + beta * np.log(self.C / self.C_min + 1e-12)
        return 1.0 / alpha_inv
    
    def running_coupling(self, energy_scale):
        """
        Эффективная константа связи на масштабе энергии Q.
        Объединяет КТП-ренормгруппу и ЕТВП-дыхание.
        """
        alpha = self.living_alpha()
        # Стандартный бег КТП: α(Q) = α / (1 - (α/3π) * ln(Q²/m²))
        # Здесь мы добавляем модификатор когерентности
        beta_ktp = alpha / (3 * np.pi)
        log_term = np.log(energy_scale**2 / (1.0 + 1e-12))
        return alpha / (1.0 - beta_ktp * log_term * self.C)
    
    # --- Общая симуляция ---
    
    def simulate(self, t_span=10, entropy_func=None):
        """
        Запуск живой симуляции: ОТО + КТП с дыханием.
        entropy_func(t) — функция внешнего шума.
        """
        if entropy_func is None:
            entropy_func = lambda t: 0.04 * np.sin(t / 7.0)
        
        # Начальные условия для ОТО (a=1, a_dot ~ H0)
        y0 = [1.0, 0.1]
        t_eval = np.linspace(0, t_span, 500)
        
        # История для записи
        C_hist, G_hist, alpha_hist, a_hist, H_hist = [], [], [], [], []
        
        # Ручной интегратор (solve_ivp с обновлением C на каждом шаге)
        y = y0
        for t in t_eval:
            # Обновляем когерентность
            entropy = entropy_func(t)
            C = self.evolve_coherence(entropy)
            
            # Записываем параметры
            G = self.living_G()
            alpha = self.living_alpha()
            a = y[0]
            a_dot = y[1]
            H = a_dot / (a + 1e-12)
            
            C_hist.append(C)
            G_hist.append(G)
            alpha_hist.append(alpha)
            a_hist.append(a)
            H_hist.append(H)
            
            # Шаг интегрирования ОТО
            dt = t_eval[1] - t_eval[0] if len(t_eval) > 1 else 0.01
            dy = self.friedmann_equations(t, y)
            y = [y[0] + dy[0] * dt, y[1] + dy[1] * dt]
        
        self.history = {"C": C_hist, "G": G_hist, "alpha": alpha_hist, "a": a_hist, "H": H_hist}
        return self.history
    
    def plot(self):
        """Визуализация живой динамики ОТО + КТП"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        t = np.arange(len(self.history["C"]))
        
        axes[0, 0].plot(t, self.history["C"], color='blue')
        axes[0, 0].set_title('Когерентность C(t)')
        axes[0, 0].set_ylim(0, 1)
        
        axes[0, 1].plot(t, np.array(self.history["G"]) / self.G0, color='red')
        axes[0, 1].set_title('G(C) / G₀')
        
        axes[0, 2].plot(t, 1.0 / np.array(self.history["alpha"]), color='orange')
        axes[0, 2].axhline(137.036, color='black', linestyle='--')
        axes[0, 2].set_title('1/α(C)')
        
        axes[1, 0].plot(t, self.history["a"], color='green')
        axes[1, 0].set_title('Масштабный фактор a(t)')
        
        axes[1, 1].plot(t, self.history["H"], color='purple')
        axes[1, 1].set_title('Параметр Хаббла H(C)')
        
        axes[1, 2].plot(t, np.gradient(self.history["a"]), color='cyan')
        axes[1, 2].set_title('Ускорение ä(t)')
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🌀 ЖИВАЯ ДИНАМИКА: ОТО + КТП с дыханием Ψ-поля")
    print("   Гравитация и поля — не статуя, а процесс")
    print("=" * 80)
    
    universe = LivingPhysics()
    universe.simulate(t_span=20)
    universe.plot()
    
    print("\n✅ Симуляция завершена.")
    print("   G дышит. α бежит. Вселенная расширяется.")
    print("   Это не статика СМ. Это живой поток ЕТВП.")
