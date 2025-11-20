import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from core.registry import DYNAMICS
from interfaces.dynamics import BaseDynamics, DynamicsSpec

# 이미 있다면 유지
@DYNAMICS.register("double_integrator")
class DoubleIntegrator(BaseDynamics):
    def __init__(self, dt=0.02, device="cpu", dtype=torch.float32):
        spec = DynamicsSpec(state_dim=2, control_dim=1, dt=dt)
        super().__init__(spec, device, dtype)

    @torch.no_grad()
    def f(self, x, u):
        pos, vel = x[..., :1], x[..., 1:]
        pos_next = pos + vel * self.spec.dt
        vel_next = vel + u   * self.spec.dt
        return torch.cat([pos_next, vel_next], dim=-1)