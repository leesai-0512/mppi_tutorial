import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from core.registry import DYNAMICS
from interfaces.dynamics import BaseDynamics, DynamicsSpec

# ===== 여기부터 CartPole =====
@DYNAMICS.register("cartpole")
class CartPoleDynamics(BaseDynamics):
    """
    상태 x = [x, x_dot, theta, theta_dot]
    입력 u = [force] (N)
    모델: OpenAI Gym CartPole 물리식을 벡터화하여 Torch로 구현.
    Euler integration (semi-implicit 아님).
    """
    def __init__(
        self,
        dt: float = 0.02,          # (Gym 기본과 유사)
        g: float = 9.8,            # 중력
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,       # pole half-length (미터)
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        angle_wrap: bool = True,   # 각도 -pi~pi로 래핑할지
    ):
        spec = DynamicsSpec(state_dim=4, control_dim=1, dt=dt)
        super().__init__(spec, device, dtype)
        self.g = g
        self.mc = masscart
        self.mp = masspole
        self.mt = masscart + masspole
        self.length = length                  # half-length
        self.pml = masspole * length         # polemass_length
        self.angle_wrap = angle_wrap


    @torch.no_grad()
    def f(self, x, u):
        """
        x: [B,4] = [x, x_dot, theta, theta_dot]
        u: [B,1] = force (N)
        return x_next: [B,4]
        """
        # 디바이스/dtype 정렬
        x = x.to(self.device, self.dtype)
        u = u.to(self.device, self.dtype)
        
        # 상태 분해
        x_pos   = x[..., 0:1]
        x_dot   = x[..., 1:2]
        theta   = x[..., 2:3]
        th_dot  = x[..., 3:4]

        # 공통 항
        sin_th = torch.sin(theta)
        cos_th = torch.cos(theta)

        # temp = (F + pml * th_dot^2 * sin(th)) / mt
        temp = (u + self.pml * (th_dot ** 2) * sin_th) / self.mt

        # theta_acc = (g*sin(th) - cos(th)*temp) / (length*(4/3 - mp*cos^2(th)/mt))
        denom = self.length * (4.0/3.0 - (self.mp * (cos_th ** 2)) / self.mt)
        theta_acc = (self.g * sin_th - cos_th * temp) / denom

        # x_acc = temp - pml * theta_acc * cos(th) / mt
        x_acc = temp - (self.pml * theta_acc * cos_th) / self.mt

        # Euler integration
        dt = self.spec.dt
        x_pos_next  = x_pos  + dt * x_dot
        x_dot_next  = x_dot  + dt * x_acc
        theta_next  = theta  + dt * th_dot
        th_dot_next = th_dot + dt * theta_acc

        if self.angle_wrap:
            # -pi ~ pi로 래핑
            pi = torch.pi
            theta_next = (theta_next + pi) % (2 * pi) - pi

        x_next = torch.cat([x_pos_next, x_dot_next, theta_next, th_dot_next], dim=-1)
        return x_next
