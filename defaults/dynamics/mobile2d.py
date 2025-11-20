import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional
from dataclasses import dataclass
import torch

from core.registry import DYNAMICS
from interfaces.dynamics import BaseDynamics, DynamicsSpec

@DYNAMICS.register("mobile2d")
class Mobile2DDynamics(BaseDynamics):
    """
    유니사이클(차량) 모델
    state x = [x, y, theta]
    control u = [v, omega]
    x_{t+1} = f(x_t, u_t):
      x' = x + dt * v * cos(theta)
      y' = y + dt * v * sin(theta)
      th'= theta + dt * omega
    """
    def __init__(self,
                 dt: float = 0.05,
                 angle_wrap: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.spec = DynamicsSpec(state_dim=3, control_dim=2, dt=dt)
        self.angle_wrap = angle_wrap

        self.device = device
        self.dtype = dtype

        # 상수 텐서화(연산 중 브로드캐스팅 편의)
        self._zero = torch.tensor(0.0, device=device, dtype=dtype)
        self._pi = torch.tensor(torch.pi, device=device, dtype=dtype)

    @torch.no_grad()
    def f(self, x, u):
        """
        x: [B,3] = [x, y, theta]
        u: [B,2] = [v, omega]
        return x_next: [B,3]
        """
        x = x.to(self.device, self.dtype)
        u = u.to(self.device, self.dtype)

        px   = x[..., 0:1]
        py   = x[..., 1:2]
        th   = x[..., 2:3]
        v    = u[..., 0:1]
        omg  = u[..., 1:2]

        dt = self.spec.dt
        cos_th = torch.cos(th)
        sin_th = torch.sin(th)

        px_next  = px + dt * v * cos_th
        py_next  = py + dt * v * sin_th
        th_next  = th + dt * omg

        if self.angle_wrap:
            # [-pi, pi] 래핑
            th_next = (th_next + self._pi) % (2 * self._pi) - self._pi

        return torch.cat([px_next, py_next, th_next], dim=-1)