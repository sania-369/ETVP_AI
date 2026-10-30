import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Literal


class ETVP_ComplexLayer(nn.Module):
    r"""
    ETVP Complex Layer v3.3 — с гиперболической упругостью.

    Изменения относительно v3.2:
      - v3.3a: добавлена гиперболическая упругость E(C) = C/(1+kC)·(1+k) как возврат к равновесию
      - v3.3b: параметр k_hyper (число Коксетера, по умолч. 30.0)
      - v3.3c: параметр gamma_restore (сила возврата, по умолч. 0.1)
      - v3.3d: возврат применяется к состоянию x перед Heun-шагом
      - v3.3e: опциональное tanh-удержание на выходе (по умолч. включено, страховка)
    """

    EPS: float = 1e-8

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 R_max: float = 2.0,
                 eta: float = 0.01,
                 beta: float = 0.1,
                 laplace_mode: Literal['auto', 'spatial', 'features', 'none'] = 'auto',
                 use_cardioid: bool = True,
                 padding_mode: Literal['reflect', 'zero', 'periodic'] = 'reflect',
                 mode: Literal['increment', 'generate'] = 'increment',
                 lambda_z: float = 1.0,
                 k_hyper: float = 30.0,
                 gamma_restore: float = 0.1,
                 use_tanh_limit: bool = True):
        super().__init__()

        if dim_in != dim_out:
            raise ValueError(
                f"ETVP_ComplexLayer требует dim_in == dim_out (получено {dim_in} и {dim_out})."
            )

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.R_max = R_max
        self.eta = eta
        self.beta = beta
        self.laplace_mode = laplace_mode
        self.use_cardioid = use_cardioid
        self.padding_mode = padding_mode
        self.mode = mode
        self.lambda_z = lambda_z
        self.k_hyper = k_hyper
        self.gamma_restore = gamma_restore
        self.use_tanh_limit = use_tanh_limit

        bound = (2.0 / dim_in) ** 0.5 / (2.0 ** 0.5)
        self.W_re = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.W_im = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.b_re = nn.Parameter(0.01 * torch.randn(dim_out))
        self.b_im = nn.Parameter(0.01 * torch.randn(dim_out))
        self.theta_mu = nn.Parameter(torch.tensor(-4.0))

    # ============================================================
    # 1. Z-ПРИНЦИП
    # ============================================================
    def get_z_penalty(self) -> torch.Tensor:
        W_norm = torch.abs(self.W_re + 1j * self.W_im).norm(dim=0)
        penalty = F.relu(W_norm - self.R_max).pow(2).mean()
        return self.lambda_z * penalty

    # ============================================================
    # 2. ГИПЕРБОЛИЧЕСКАЯ УПРУГОСТЬ (возврат к равновесию)
    # ============================================================
    def hyperbolic_restore(self, re: torch.Tensor, im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Гиперболическая упругость:
            E(C) = C / (1 + k·C) · (1 + k)
        где C — нормированная амплитуда (0..1), k — число Коксетера.

        Возвращает силу, тянущую состояние к равновесию при |x| > 0.
        При |x| → 0 сила → 0 (нет сингулярности).
        При |x| → ∞ сила насыщается (не даёт взрыва).
        """
        mag = torch.sqrt(re**2 + im**2 + self.EPS)
        # Нормируем амплитуду в [0, 1] относительно R_max
        C = mag / self.R_max
        # Гиперболическая упругость: E(C) = C / (1 + k·C) · (1 + k)
        E = C / (1.0 + self.k_hyper * C) * (1.0 + self.k_hyper)
        # Сила возврата: gamma * (E - C) * direction
        # При C > E — тормозит; при C < E — не тянет (чтобы не разгонять)
        force = self.gamma_restore * (E - C)
        # Направление: против движения (к нулю)
        force_re = -force * re / mag
        force_im = -force * im / mag
        return force_re, force_im

    # ============================================================
    # 3. ЛАПЛАСИАН
    # ============================================================
    def laplacian_spatial(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4:
            return torch.zeros_like(z)
        _, _, H, W = z.shape
        if H < 2 or W < 2:
            return torch.zeros_like(z)
        pad_map = {'reflect': 'reflect', 'zero': 'constant', 'periodic': 'circular'}
        z_padded = F.pad(z, (1, 1, 1, 1), mode=pad_map.get(self.padding_mode, 'reflect'))
        lap = (z_padded[:, :, :-2, 1:-1] +
               z_padded[:, :, 2:, 1:-1] +
               z_padded[:, :, 1:-1, :-2] +
               z_padded[:, :, 1:-1, 2:] -
               4 * z)
        return lap / (H * W + self.EPS)

    def laplacian(self, z: torch.Tensor) -> torch.Tensor:
        mode = self.laplace_mode
        if mode == 'auto':
            mode = 'spatial' if z.dim() == 4 else 'none'
        if mode == 'spatial':
            return self.laplacian_spatial(z)
        return torch.zeros_like(z)

    # ============================================================
    # 4. КОМПЛЕКСНОЕ УМНОЖЕНИЕ И АКТИВАЦИЯ
    # ============================================================
    def complex_mul(self, x_re: torch.Tensor, x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_re.dim() == 2:
            out_re = x_re @ self.W_re - x_im @ self.W_im + self.b_re
            out_im = x_re @ self.W_im + x_im @ self.W_re + self.b_im
        elif x_re.dim() == 4:
            out_re = (torch.einsum('bihw,io->bohw', x_re, self.W_re) -
                      torch.einsum('bihw,io->bohw', x_im, self.W_im) +
                      self.b_re.view(1, -1, 1, 1))
            out_im = (torch.einsum('bihw,io->bohw', x_re, self.W_im) +
                      torch.einsum('bihw,io->bohw', x_im, self.W_re) +
                      self.b_im.view(1, -1, 1, 1))
        else:
            raise ValueError(f"Unsupported tensor dims: {x_re.dim()}.")
        return out_re, out_im

    def cardioid_activation(self, re: torch.Tensor, im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        angle = torch.atan2(im, re)
        scale = 0.5 * (1.0 + torch.cos(angle))
        return scale * re, scale * im

    # ============================================================
    # 5. ОПЕРАТОР ЭВОЛЮЦИИ
    # ============================================================
    def forward(self,
                x_re: torch.Tensor,
                x_im: torch.Tensor,
                dt: float = 0.1,
                return_intermediate: bool = False):
        # --- Возврат к равновесию перед эволюцией (v3.3) ---
        restore_re, restore_im = self.hyperbolic_restore(x_re, x_im)

        # --- Шаг 1: k1 ---
        f_re, f_im = self.complex_mul(x_re, x_im)
        if self.use_cardioid:
            f_re, f_im = self.cardioid_activation(f_re, f_im)
        else:
            f_re, f_im = torch.tanh(f_re), torch.tanh(f_im)

        mu = F.softplus(self.theta_mu)
        lap_re = self.laplacian(f_re)
        lap_im = self.laplacian(f_im)

        f_abs = torch.sqrt(f_re**2 + f_im**2) + self.EPS
        C = torch.tanh(f_abs.mean()).item()

        sqrt_dt = math.sqrt(dt)
        noise_scale = self.eta * torch.tanh(self.beta * f_abs)
        phase = torch.atan2(f_im, f_re)
        noise_re = sqrt_dt * noise_scale * torch.randn_like(f_re) * torch.cos(phase)
        noise_im = sqrt_dt * noise_scale * torch.randn_like(f_im) * torch.sin(phase)

        k1_re = f_re - mu * lap_re
        k1_im = f_im - mu * lap_im
        x_pred_re = x_re + dt * k1_re
        x_pred_im = x_im + dt * k1_im

        # --- Шаг 2: k2 ---
        f_pred_re, f_pred_im = self.complex_mul(x_pred_re, x_pred_im)
        if self.use_cardioid:
            f_pred_re, f_pred_im = self.cardioid_activation(f_pred_re, f_pred_im)
        else:
            f_pred_re, f_pred_im = torch.tanh(f_pred_re), torch.tanh(f_pred_im)

        lap_pred_re = self.laplacian(f_pred_re)
        lap_pred_im = self.laplacian(f_pred_im)
        k2_re = f_pred_re - mu * lap_pred_re
        k2_im = f_pred_im - mu * lap_pred_im

        # --- Шаг 3: Heun + возврат + шум ---
        z_re_next = x_re + dt * (k1_re + k2_re) / 2 + dt * restore_re + noise_re
        z_im_next = x_im + dt * (k1_im + k2_im) / 2 + dt * restore_im + noise_im

        # --- Страховка: tanh-удержание (v3.3e) ---
        if self.use_tanh_limit:
            z_re_next = self.R_max * torch.tanh(z_re_next / self.R_max)
            z_im_next = self.R_max * torch.tanh(z_im_next / self.R_max)

        if return_intermediate:
            return z_re_next, z_im_next, {
                'f': (f_re, f_im),
                'f_pred': (f_pred_re, f_pred_im),
                'restore': (restore_re, restore_im),
                'mu': mu,
                'C': C,
                'dt': dt
            }

        return z_re_next, z_im_next
