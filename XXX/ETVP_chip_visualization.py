#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌀 ETVP Si-Photon Core X — 3D визуализация чипа
Отображает все слои архитектуры:
- Слой 0: 256 колец Si₃N₄ (синие)
- Слой 1: pn-переходы (красные) для модуляции C
- Слой 2: JPA-массив (зелёные) для FFS-шума
- Слой 3: MEMS-переключатель + спираль памяти (жёлтые)

Управление: вращение, масштабирование, панорамирование.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# 1. ГЕНЕРАЦИЯ ГЕОМЕТРИИ ЧИПА
# =============================================================================

class ChipVisualizer:
    def __init__(self):
        self.Phi = (1 + np.sqrt(5)) / 2
        self.num_rings = 256
        self.radii = []
        self.positions = []
        self.generate_rings()

    def generate_rings(self):
        """Генерация 256 колец с радиусами по Phi и позициями по спирали."""
        R_min = 10  # мкм
        for i in range(self.num_rings):
            R = R_min * (self.Phi ** (i / 16.0))
            theta = i * 0.05
            x = 500 + 15 * i * np.cos(theta)  # мкм
            y = 500 + 15 * i * np.sin(theta)
            self.radii.append(R)
            self.positions.append((x, y))
        return self.radii, self.positions

    def create_3d_matplotlib(self):
        """Визуализация в matplotlib (3D)."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # --- Слой 0: Кольца (синие) ---
        for i, (x, y) in enumerate(self.positions):
            r = self.radii[i]
            # Круг в 3D (z=0)
            theta = np.linspace(0, 2*np.pi, 50)
            x_circle = x + r * np.cos(theta)
            y_circle = y + r * np.sin(theta)
            z_circle = np.zeros_like(theta)
            ax.plot(x_circle, y_circle, z_circle, color='blue', alpha=0.3, linewidth=0.5)

        # --- Слой 1: pn-переходы (красные, z=1) ---
        for i, (x, y) in enumerate(self.positions):
            # Спираль нагревателя (упрощённо: маленький квадрат)
            size = 0.5
            ax.scatter(x, y, 1, color='red', s=5, alpha=0.4)

        # --- Слой 2: JPA-массив (зелёные, z=2) ---
        for j in range(8):
            for k in range(8):
                x_jpa = 100 + j * 80
                y_jpa = 100 + k * 80
                ax.scatter(x_jpa, y_jpa, 2, color='green', s=20, alpha=0.7)

        # --- Слой 3: MEMS + спираль (жёлтые, z=3) ---
        # Спиральный шлейф (упрощённо: спираль)
        spiral_x, spiral_y = [], []
        for t in np.linspace(0, 30*np.pi, 1000):
            r_spiral = 5 + t * 0.1
            spiral_x.append(800 + r_spiral * np.cos(t))
            spiral_y.append(800 + r_spiral * np.sin(t))
        ax.plot(spiral_x, spiral_y, 3, color='gold', linewidth=2, alpha=0.9)
        
        # MEMS-переключатель (кантилеввер)
        ax.plot([780, 820], [800, 800], 3, color='orange', linewidth=4, alpha=0.9)
        ax.scatter(800, 800, 3, color='red', s=50, marker='^')

        # --- Настройка ---
        ax.set_xlabel('X (мкм)', fontsize=10)
        ax.set_ylabel('Y (мкм)', fontsize=10)
        ax.set_zlabel('Z (слой)', fontsize=10)
        ax.set_title('ETVP Si-Photon Core X — 3D вид\n(Синий: кольца, Красный: pn, Зелёный: JPA, Жёлтый: память)', fontsize=12)
        ax.view_init(elev=30, azim=45)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_zlim(-1, 5)

        plt.tight_layout()
        plt.savefig('chip_3d_matplotlib.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_plotly_3d(self):
        """Интерактивная визуализация в Plotly (HTML)."""
        fig = go.Figure()

        # --- Слой 0: Кольца (синие) ---
        for i, (x, y) in enumerate(self.positions):
            r = self.radii[i]
            theta = np.linspace(0, 2*np.pi, 50)
            x_circle = x + r * np.cos(theta)
            y_circle = y + r * np.sin(theta)
            z_circle = np.zeros_like(theta)
            
            fig.add_trace(go.Scatter3d(
                x=x_circle, y=y_circle, z=z_circle,
                mode='lines',
                line=dict(color='blue', width=1, opacity=0.3),
                name='Кольца Si₃N₄',
                showlegend=False
            ))

        # --- Слой 1: pn-переходы (красные, z=1) ---
        for i, (x, y) in enumerate(self.positions):
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[1],
                mode='markers',
                marker=dict(size=2, color='red', opacity=0.4),
                name='pn-переходы',
                showlegend=False
            ))

        # --- Слой 2: JPA-массив (зелёные, z=2) ---
        jpa_x, jpa_y = [], []
        for j in range(8):
            for k in range(8):
                jpa_x.append(100 + j * 80)
                jpa_y.append(100 + k * 80)
        fig.add_trace(go.Scatter3d(
            x=jpa_x, y=jpa_y, z=[2]*len(jpa_x),
            mode='markers',
            marker=dict(size=5, color='green', opacity=0.8, symbol='square'),
            name='JPA',
            showlegend=True
        ))

        # --- Слой 3: MEMS + спираль (жёлтые, z=3) ---
        # Спираль
        spiral_x, spiral_y = [], []
        for t in np.linspace(0, 30*np.pi, 500):
            r_spiral = 5 + t * 0.1
            spiral_x.append(800 + r_spiral * np.cos(t))
            spiral_y.append(800 + r_spiral * np.sin(t))
        fig.add_trace(go.Scatter3d(
            x=spiral_x, y=spiral_y, z=[3]*len(spiral_x),
            mode='lines',
            line=dict(color='gold', width=3, opacity=0.9),
            name='Спираль памяти',
            showlegend=True
        ))
        
        # MEMS-переключатель
        fig.add_trace(go.Scatter3d(
            x=[780, 820], y=[800, 800], z=[3, 3],
            mode='lines+markers',
            line=dict(color='orange', width=6),
            marker=dict(size=10, color='red', symbol='triangle-up'),
            name='MEMS',
            showlegend=True
        ))

        # --- Настройка ---
        fig.update_layout(
            title='ETVP Si-Photon Core X — Интерактивная 3D-модель<br>Синий: кольца (слой 0), Красный: pn (слой 1), Зелёный: JPA (слой 2), Жёлтый: память (слой 3)',
            scene=dict(
                xaxis_title='X (мкм)',
                yaxis_title='Y (мкм)',
                zaxis_title='Z (слой)',
                xaxis_range=[0, 1000],
                yaxis_range=[0, 1000],
                zaxis_range=[-1, 5],
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1000,
            height=800,
            showlegend=True,
            legend=dict(x=0.8, y=0.9)
        )

        # Сохранение в HTML
        fig.write_html('chip_3d_plotly.html')
        print("✅ Интерактивная 3D-модель сохранена как 'chip_3d_plotly.html'")
        fig.show()

    def create_2d_top_view(self):
        """2D-вид сверху (для GDSII-подобного отображения)."""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Кольца
        for i, (x, y) in enumerate(self.positions):
            r = self.radii[i]
            circle = plt.Circle((x, y), r, color='blue', alpha=0.2, linewidth=0.5, fill=False)
            ax.add_patch(circle)
        
        # pn-переходы (красные точки)
        for i, (x, y) in enumerate(self.positions):
            ax.scatter(x, y, color='red', s=3, alpha=0.5)
        
        # JPA-массив (зелёные квадраты)
        for j in range(8):
            for k in range(8):
                x_jpa = 100 + j * 80
                y_jpa = 100 + k * 80
                ax.scatter(x_jpa, y_jpa, color='green', s=50, marker='s', alpha=0.7)
        
        # Спираль памяти
        spiral_x, spiral_y = [], []
        for t in np.linspace(0, 30*np.pi, 1000):
            r_spiral = 5 + t * 0.1
            spiral_x.append(800 + r_spiral * np.cos(t))
            spiral_y.append(800 + r_spiral * np.sin(t))
        ax.plot(spiral_x, spiral_y, color='gold', linewidth=2, alpha=0.9)
        
        # MEMS
        ax.plot([780, 820], [800, 800], color='orange', linewidth=4)
        ax.scatter(800, 800, color='red', s=100, marker='^')
        
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_aspect('equal')
        ax.set_title('ETVP Si-Photon Core X — Вид сверху\n(Синий: кольца, Красный: pn, Зелёный: JPA, Жёлтый: память)', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('chip_2d_topview.png', dpi=300, bbox_inches='tight')
        plt.show()

# =============================================================================
# 2. ЗАПУСК ВИЗУАЛИЗАЦИИ
# =============================================================================

if __name__ == "__main__":
    print("🌀 Генерация 3D-визуализации ETVP Si-Photon Core X...")
    viz = ChipVisualizer()
    
    # Генерация геометрии
    print(f"   Сгенерировано {viz.num_rings} колец.")
    
    # Matplotlib 3D
    print("   Создание 3D-вида (matplotlib)...")
    viz.create_3d_matplotlib()
    
    # Plotly интерактивный
    print("   Создание интерактивной 3D-модели (plotly)...")
    viz.create_plotly_3d()
    
    # Вид сверху
    print("   Создание вида сверху...")
    viz.create_2d_top_view()
    
    print("\n✅ Визуализация завершена. Созданы файлы:")
    print("   - chip_3d_matplotlib.png (статичный 3D)")
    print("   - chip_3d_plotly.html (интерактивный 3D)")
    print("   - chip_2d_topview.png (вид сверху)")
    print("\nОткройте 'chip_3d_plotly.html' в браузере для вращения и масштабирования.")
