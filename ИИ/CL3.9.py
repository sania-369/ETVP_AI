import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Literal, Optional


class ETVP_ComplexLayer(nn.Module):
    r"""
    ETVP Complex Layer v3.9 — C как состояние, управляющее режимом.

    Изменения относительно v3.8:
      - v3.9a: C_state — состояние когерентности с релаксацией (dC/dt)
      - v3.9b: C выбирает режим: высокая C — эволюция, низкая C — равновесие
      - v3.9c: шум НЕ модулируется C (шум в среде, C — ответ на него)
      - v3.9d: C_history для отслеживания траектории
      - v3.9e: методы get_last_C(), reset_C()

    Логика: C — мера удержания структуры. Шум есть всегда.
    Высокая C — структура держится, эволюция идёт.
    Низкая C — структура распадается, упругость тянет к равновесию.
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
                 use_tanh_limit: bool = True,
                 freeze_theta_mu: bool = True,
                 gamma_c: float = 0.1,
                 C_FFS: float = 1.0,
                 C_history_size: int = 100):
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
        self.freeze_theta_mu = freeze_theta_mu
        self.gamma_c = gamma_c
        self.C_FFS = C_FFS
        self.C_history_size = C_history_size

        bound = (2.0 / dim_in) ** 0.5 / (2.0 ** 0.5)
        self.W_re = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.W_im = nn.Parameter(torch.randn(dim_in, dim_out) * bound)
        self.b_re = nn.Parameter(0.01 * torch.randn(dim_out))
        self.b_im = nn.Parameter(0.01 * torch.randn(dim_out))
        self.theta_mu = nn.Parameter(torch.tensor(-4.0))
        if freeze_theta_mu:
            self.theta_mu.requires_grad = False

        # C как состояние (не параметр — не обучается напрямую)
        self.register_buffer('C_state', torch.tensor(0.5))
        self.C_history = []

    def get_param_groups(self, weight_decay: float = 1e-3):
        wd_params = [self.W_re, self.W_im]
        no_wd_params = [self.b_re, self.b_im]
        if not self.freeze_theta_mu:
            wd_params.append(self.theta_mu)
        return [
            {'params': wd_params, 'weight_decay': weight_decay},
            {'params': no_wd_params, 'weight_decay': 0.0},
        ]

    def get_z_penalty(self) -> torch.Tensor:
        W_norm = torch.abs(self.W_re + 1j * self.W_im).norm(dim=0)
        penalty = F.relu(W_norm - self.R_max).pow(2).mean()
        return self.lambda_z * penalty

    def hyperbolic_restore(self, re: torch.Tensor, im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag = torch.sqrt(re**2 + im**2 + 1e-6)
        C = mag / self.R_max
        E = C / (1.0 + self.k_hyper * C) * (1.0 + self.k_hyper)
        force = -self.gamma_restore * (C - E)
        return force * re, force * im

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
        if mode == 'spatial' and z.dim() == 4:
            return self.laplacian_spatial(z)
        return torch.zeros_like(z)

    def complex_mul(self, x_re: torch.Tensor, x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dims = x_re.dim()
        if dims == 2:
            out_re = x_re @ self.W_re - x_im @ self.W_im + self.b_re
            out_im = x_re @ self.W_im + x_im @ self.W_re + self.b_im
        elif dims == 3:
            out_re = x_re @ self.W_re - x_im @ self.W_im + self.b_re
            out_im = x_re @ self.W_im + x_im @ self.W_re + self.b_im
        elif dims == 4:
            out_re = (torch.einsum('bihw,io->bohw', x_re, self.W_re) -
                      torch.einsum('bihw,io->bohw', x_im, self.W_im) +
                      self.b_re.view(1, -1, 1, 1))
            out_im = (torch.einsum('bihw,io->bohw', x_re, self.W_im) +
                      torch.einsum('bihw,io->bohw', x_im, self.W_re) +
                      self.b_im.view(1, -1, 1, 1))
        else:
            raise ValueError(f"Unsupported tensor dims: {dims}. Expected 2D, 3D or 4D.")
        return out_re, out_im

    def cardioid_activation(self, re: torch.Tensor, im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = torch.sigmoid(re)
        return scale * re, scale * im

    def update_C(self, C_new: torch.Tensor) -> torch.Tensor:
        """
        Обновление C_state с релаксацией (как dC/dt в Мат_аппарат):
            C_state = C_state*(1-γ) + C_new*γ
        C_new — мгновенная когерентность от текущего состояния.
        """
        C_new_scalar = C_new.detach().mean()
        self.C_state = self.C_state * (1 - self.gamma_c) + C_new_scalar * self.gamma_c
        self.C_state = torch.clamp(self.C_state, 0.0, 1.0)
        self.C_history.append(self.C_state.item())
        if len(self.C_history) > self.C_history_size:
            self.C_history.pop(0)
        return self.C_state

    def get_last_C(self) -> Optional[float]:
        return self.C_state.item() if self.C_state is not None else None

    def reset_C(self):
        self.C_state = torch.tensor(0.5, device=self.C_state.device)
        self.C_history = []

    def forward(self,
                x_re: torch.Tensor,
                x_im: torch.Tensor,
                dt: float = 0.1,
                return_intermediate: bool = False):
        restore_re, restore_im = self.hyperbolic_restore(x_re, x_im)

        f_re, f_im = self.complex_mul(x_re, x_im)
        if self.use_cardioid:
            f_re, f_im = self.cardioid_activation(f_re, f_im)
        else:
            f_re, f_im = torch.tanh(f_re), torch.tanh(f_im)

        mu = F.softplus(self.theta_mu)
        lap_re = self.laplacian(f_re)
        lap_im = self.laplacian(f_im)

        f_abs = torch.sqrt(f_re**2 + f_im**2) + self.EPS
        C_new = torch.tanh(f_abs.mean())
        C_state = self.update_C(C_new)

        # Шум НЕ модулируется C (шум в среде)
        sqrt_dt = math.sqrt(dt)
        noise_scale = self.eta * torch.tanh(self.beta * f_abs)
        f_mag = torch.sqrt(f_re**2 + f_im**2 + 1e-6)
        noise_re = sqrt_dt * noise_scale * torch.randn_like(f_re) * (f_re / f_mag)
        noise_im = sqrt_dt * noise_scale * torch.randn_like(f_im) * (f_im / f_mag)

        k1_re = f_re - mu * lap_re
        k1_im = f_im - mu * lap_im
        x_pred_re = x_re + dt * k1_re
        x_pred_im = x_im + dt * k1_im

        f_pred_re, f_pred_im = self.complex_mul(x_pred_re, x_pred_im)
        if self.use_cardioid:
            f_pred_re, f_pred_im = self.cardioid_activation(f_pred_re, f_pred_im)
        else:
            f_pred_re, f_pred_im = torch.tanh(f_pred_re), torch.tanh(f_pred_im)

        lap_pred_re = self.laplacian(f_pred_re)
        lap_pred_im = self.laplacian(f_pred_im)
        k2_re = f_pred_re - mu * lap_pred_re
        k2_im = f_pred_im - mu * lap_pred_im

        # Эволюция (структура держится) и равновесие (структура распалась)
        evolve_re = x_re + dt * (k1_re + k2_re) / 2 + noise_re
        evolve_im = x_im + dt * (k1_im + k2_im) / 2 + noise_im
        equil_re = x_re + dt * restore_re
        equil_im = x_im + dt * restore_im

        # C выбирает режим (v3.9b)
        z_re_next = C_state * evolve_re + (1 - C_state) * equil_re
        z_im_next = C_state * evolve_im + (1 - C_state) * equil_im

        if self.use_tanh_limit:
            z_re_next = self.R_max * torch.tanh(z_re_next / self.R_max)
            z_im_next = self.R_max * torch.tanh(z_im_next / self.R_max)

        if return_intermediate:
            return z_re_next, z_im_next, {
                'f': (f_re, f_im),
                'f_pred': (f_pred_re, f_pred_im),
                'restore': (restore_re, restore_im),
                'mu': mu,
                'C': C_state.item(),
                'C_new': C_new.item(),
                'dt': dt
            }
        return z_re_next, z_im_next
