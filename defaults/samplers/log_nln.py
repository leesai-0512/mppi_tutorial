import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from core.registry import SAMPLERS
from interfaces.sampler import BaseNoiseSampler

@SAMPLERS.register("log_nln")
class LogNLNSampler(BaseNoiseSampler):
    """
    Z = X * exp(W)
    X ~ N(0, σ_n^2),   W ~ N(μ_ln, σ_ln^2)
    - target_std (σ_u)을 주면 μ_ln을 자동 계산해 최종 분산 Var[Z]=σ_u^2 로 맞춤
    - μ_ln을 직접 지정할 수도 있음(둘 다 주면 μ_ln을 우선함)
    - control_dim 별 개별 파라미터 벡터 지원
    - center_batch=True면 배치 평균을 0으로 센터링(소표본 안정화용, 선택)
    """
    def __init__(self,
                 control_dim, horizon,
                 std_n=0.045,             # σ_n
                 std_ln=0.22,             # σ_ln
                 mu_ln=None,              # 지정 시 그대로 사용
                 target_std=None,         # σ_u (지정 시 μ_ln 자동계산)
                 center_batch=False,      # 배치 평균 0 강제 센터링(옵션)
                 device="cpu", dtype=torch.float32):
        super().__init__(control_dim, horizon)
        self.device, self.dtype = device, dtype
        self.center_batch = center_batch

        def as_vec(x, name):
            if x is None: return None
            if isinstance(x, (float, int)):
                return torch.full((control_dim,), float(x), device=device, dtype=dtype)
            t = torch.as_tensor(x, device=device, dtype=dtype)
            if t.numel() == 1: t = t.expand(control_dim)
            assert t.shape == (control_dim,), f"{name} must have shape (control_dim,)"
            return t

        self.std_n  = as_vec(std_n,  "std_n")
        self.std_ln = as_vec(std_ln, "std_ln")

        mu_vec      = as_vec(mu_ln,  "mu_ln") if mu_ln is not None else None
        target_vec  = as_vec(target_std, "target_std") if target_std is not None else None

        # μ_ln 자동 계산: σ_u^2 = σ_n^2 * exp(2μ_ln + 2σ_ln^2)
        if mu_vec is None and target_vec is not None:
            eps = torch.finfo(self.dtype).eps
            ratio = torch.clamp((target_vec**2) / (self.std_n**2 + eps), min=eps)
            self.mu_ln = 0.5 * (torch.log(ratio) - 2.0*(self.std_ln**2))
        else:
            # μ_ln 직접 지정 또는 0 기본값
            self.mu_ln = mu_vec if mu_vec is not None else torch.zeros(control_dim, device=device, dtype=dtype)

    def sample(self, B: int):
        shape = (B, self.horizon, self.control_dim)
        std_n  = self.std_n.view(1,1,-1)
        std_ln = self.std_ln.view(1,1,-1)
        mu_ln  = self.mu_ln.view(1,1,-1)

        X = torch.randn(shape, device=self.device, dtype=self.dtype) * std_n
        W = torch.randn(shape, device=self.device, dtype=self.dtype) * std_ln + mu_ln
        Z = X * torch.exp(W)

        if self.center_batch:
            Z = Z - Z.mean(dim=0, keepdim=True)
        return Z
