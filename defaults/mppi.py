import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from dataclasses import dataclass
from core.registry import ALGOS
from interfaces.dynamics import BaseDynamics
from interfaces.cost import BaseCost
from interfaces.sampler import BaseNoiseSampler

@dataclass
class MPPIConfig:
    horizon: int = 20
    samples: int = 1024
    lambda_: float = 1.0
    gamma: float = 1.0
    u_clip: torch.Tensor | None = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    record_sample: bool = False

@ALGOS.register("mppi")
class MPPI:
    def __init__(self, dynamics: BaseDynamics, cost: BaseCost, sampler: BaseNoiseSampler, cfg: MPPIConfig):
        self.f = dynamics
        self.c = cost
        self.sampler = sampler
        self.cfg = cfg
        T, U = cfg.horizon, dynamics.spec.control_dim
        self.u_seq = torch.zeros(T, U, device=cfg.device, dtype=cfg.dtype)

        if cfg.u_clip is not None:
            if isinstance(cfg.u_clip, (float, int)):
                cfg.u_clip = torch.tensor([cfg.u_clip] * U, device=cfg.device, dtype=cfg.dtype)
            elif isinstance(cfg.u_clip, (list, tuple)):
                cfg.u_clip = torch.tensor(cfg.u_clip, device=cfg.device, dtype=cfg.dtype)
            elif isinstance(cfg.u_clip, torch.Tensor):
                cfg.u_clip = cfg.u_clip.to(device=cfg.device, dtype=cfg.dtype)
            else:
                raise TypeError(f"Unsupported type for u_clip: {type(cfg.u_clip)}")

        if self.cfg.gamma != 1.0:
            self.gamma = torch.pow(
                torch.as_tensor(cfg.gamma, device=cfg.device, dtype=cfg.dtype),
                torch.arange(T, device=cfg.device, dtype=cfg.dtype)
            ).view(1, T)                                  # [1,T]
        else:
            self.gamma = cfg.gamma

        if cfg.record_sample is None:
            self.record_sample = False
        else:
            self.record_sample = cfg.record_sample


    @torch.no_grad()
    def rollout(self, x0):
        B, T, U = self.cfg.samples, self.cfg.horizon, self.f.spec.control_dim
        device, dtype = self.cfg.device, self.cfg.dtype
        S = torch.zeros(B, device=device, dtype=dtype)
        Xs = torch.empty(B, T+1, self.f.spec.state_dim, device=device, dtype=dtype)
        noise = self.sampler.sample(B)                               # [B,T,U]
        u = self.u_seq.unsqueeze(0) + noise                          # [B,T,U]

        if self.cfg.u_clip is not None:
                umin = -self.cfg.u_clip.view(1, -1)  # [1,U]
                umax =  self.cfg.u_clip.view(1, -1)  # [1,U]
                u = torch.clamp(u, min=umin, max=umax)
                noise = u - self.u_seq
        
        x = x0.expand(B, -1).to(device=device, dtype=dtype)          # [B,S]
        Xs[:, 0] = x

        for t in range(T):
            ut = u[:, t, :]
            x = self.f.step(x, ut)
            Xs[:, t+1] = x

        X = Xs[:, :-1]                                      # [B,T,S]
        C_stage = self.c.stage(X, u) * self.gamma               # [B,T]
        S = C_stage.sum(dim=1) + self.c.terminal(Xs[:, -1])
        return S, noise, Xs

    @torch.no_grad()
    def weights(self, costs):
        c_min = torch.min(costs)
        w = torch.exp(-(costs-c_min) / self.cfg.lambda_)
        w = w / (w.sum() + 1e-12)
        return w

    @torch.no_grad()
    def update(self, x0):
        costs, noise, Xs = self.rollout(x0)
        w = self.weights(costs)                         # [B]
        delta = (w.view(-1,1,1) * noise).sum(dim=0)      # [T,U]
        self.u_seq = self.u_seq + delta
        return self.u_seq[0], Xs , noise                             # 첫 제어

    @torch.no_grad()
    def step(self, x_now):
        u0, Xs, noise = self.update(x_now)
        self.u_seq = torch.cat([self.u_seq[1:], self.u_seq[-1:]], dim=0)
        if self.record_sample:
            return u0, Xs, noise
        else:
            return u0
