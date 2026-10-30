import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Literal


class ETVP_ComplexLayer(nn.Module):
    r"""
    ETVP Complex Layer v3.2 — рабочая версия.

    Изменения относительно v3.0:
      - v3.2a: assert dim_in == dim_out (смена размерности — снаружи через nn.Linear)
      - v3.2b: theta_mu = -4.0 (mu ≈ 0.018 на старте)
      - v3.2c: возврат когерентности C в intermediates
      - v3.2d: Xavier поделён на sqrt(2) для комплексных весов
      - v3.2e: защита от H<2 / W<2 в лапласиане
    """

    EPS: float = 1e-8

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 R_max: float = 1.0,
                 eta: float = 0.01,
                 beta: float = 0.1,
                 laplace_mode: Literal['auto', 'spatial', 'features', 'none'] = 'auto',
                 use_cardioid: bool = True,
                 padding_mode: Literal['reflect', 'zero', 'periodic'] = 'reflect',
                 mode: Literal['increment', 'generate'] = 'increment',
                 lambda_z: float = 1.0):
        super().__init__()

        if dim_in != dim_out:
            raise ValueError(
                f"ETVP_ComplexLayer требует dim_in == dim_out (получено {dim_in} и {dim_out}). "
                f"Для смены размерности используйте nn.Linear перед слоем."
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

        # --- Комплексные веса (Xavier, делим на sqrt(2)) ---
        bound = (2.0 / dim_in) ** 0.5 / (2.0 ** 0.5)
        self.W_re = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.W_im = nn.Parameter(torch.randn(dim_in, dim_out) * bound)

        self.b_re = nn.Parameter(0.01 * torch.randn(dim_out))
        self.b_im = nn.Parameter(0.01 * torch.randn(dim_out))

        # --- Обучаемая вязкость (mu ≈ 0.018 на старте) ---
        self.theta_mu = nn.Parameter(torch.tensor(-4.0))

    # ============================================================
    # 1. Z-ПРИНЦИП
    # ============================================================
    def get_z_penalty(self) -> torch.Tensor:
        W_norm = torch.abs(self.W_re + 1j * self.W_im).norm(dim=0)
        penalty = F.relu(W_norm - self.R_max).pow(2).mean()
        return self.lambda_z * penalty

    # ============================================================
    # 2. ЛАПЛАСИАН
    # ============================================================
    def laplacian_spatial(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4:
            return torch.zeros_like(z)
        _, _, H, W = z.shape
        if H < 2 or W < 2:
            return torch.zeros_like(z)
        pad_map = {'reflect': 'reflect', 'zero': 'constant', 'periodic': 'circular'}
        pad_mode = pad_map.get(self.padding_mode, 'reflect')
        z_padded = F.pad(z, (1, 1, 1, 1), mode=pad_mode)
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
    # 3. КОМПЛЕКСНОЕ УМНОЖЕНИЕ И АКТИВАЦИЯ
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
            raise ValueError(f"Unsupported tensor dims: {x_re.dim()}. Expected 2D or 4D.")
        return out_re, out_im

    def cardioid_activation(self, re: torch.Tensor, im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        angle = torch.atan2(im, re)
        scale = 0.5 * (1.0 + torch.cos(angle))
        return scale * re, scale * im

    # ============================================================
    # 4. ОПЕРАТОР ЭВОЛЮЦИИ
    # ============================================================
    def forward(self,
                x_re: torch.Tensor,
                x_im: torch.Tensor,
                dt: float = 0.1,
                return_intermediate: bool = False):
        # --- Шаг 1: k1 ---
        f_re, f_im = self.complex_mul(x_re, x_im)
        if self.use_cardioid:
            f_re, f_im = self.cardioid_activation(f_re, f_im)
        else:
            f_re, f_im = torch.tanh(f_re), torch.tanh(f_im)

        mu = F.softplus(self.theta_mu)
        lap_re = self.laplacian(f_re)
        lap_im = self.laplacian(f_im)

        # Когерентность (диагностика)
        f_abs = torch.sqrt(f_re**2 + f_im**2) + self.EPS
        C = torch.tanh(f_abs.mean()).item()

        sqrt_dt = math.sqrt(dt)
        noise_scale = self.eta * torch.tanh(self.beta * f_abs)
        phase = torch.atan2(f_im, f_re)
        noise_re = sqrt_dt * noise_scale * torch.randn_like(f_re) * torch.cos(phase)
        noise_im = sqrt_dt * noise_scale * torch.randn_like(f_im) * torch.sin(phase)

        if self.mode == 'increment':
            k1_re = f_re - mu * lap_re
            k1_im = f_im - mu * lap_im
            x_pred_re = x_re + dt * k1_re
            x_pred_im = x_im + dt * k1_im
        else:  # generate
            k1_re = f_re - mu * lap_re
            k1_im = f_im - mu * lap_im
            x_pred_re = f_re - dt * mu * lap_re
            x_pred_im = f_im - dt * mu * lap_im

        # --- Шаг 2: k2 ---
        f_pred_re, f_pred_im = self.complex_mul(x_pred_re, x_pred_im)
        if self.use_cardioid:
            f_pred_re, f_pred_im = self.cardioid_activation(f_pred_re, f_pred_im)
        else:
            f_pred_re, f_pred_im = torch.tanh(f_pred_re), torch.tanh(f_pred_im)

        lap_pred_re = self.laplacian(f_pred_re)
        lap_pred_im = self.laplacian(f_pred_im)

        if self.mode == 'increment':
            k2_re = f_pred_re - mu * lap_pred_re
            k2_im = f_pred_im - mu * lap_pred_im
        else:
            k2_re = f_pred_re - mu * lap_pred_re
            k2_im = f_pred_im - mu * lap_pred_im

        # --- Шаг 3: Heun ---
        if self.mode == 'increment':
            z_re_next = x_re + dt * (k1_re + k2_re) / 2 + noise_re
            z_im_next = x_im + dt * (k1_im + k2_im) / 2 + noise_im
        else:
            z_re_next = (f_re + f_pred_re) / 2 - dt * mu * (lap_re + lap_pred_re) / 2
            z_im_next = (f_im + f_pred_im) / 2 - dt * mu * (lap_im + lap_pred_im) / 2

        if return_intermediate:
            return z_re_next, z_im_next, {
                'f': (f_re, f_im),
                'f_pred': (f_pred_re, f_pred_im),
                'lap': (lap_re, lap_im),
                'noise': (noise_re, noise_im),
                'mu': mu,
                'C': C,
                'dt': dt
            }

        return z_re_next, z_im_next
