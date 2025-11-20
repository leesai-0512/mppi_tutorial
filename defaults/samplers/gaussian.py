import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from core.registry import SAMPLERS
from interfaces.sampler import BaseNoiseSampler

@SAMPLERS.register("gaussian")
class GaussianSampler(BaseNoiseSampler):
    def __init__(self, control_dim, horizon, std_init=0.5, device="cpu", dtype=torch.float32):
        super().__init__(control_dim, horizon)
        self.std = std_init
        self.device = device
        self.dtype = dtype

        # std_init이 스칼라면 모든 control에 동일하게, 리스트/텐서면 그대로 사용
        if isinstance(std_init, (float, int)):
            self.std = torch.full((control_dim,), std_init, device=device, dtype=dtype)
        else:
            self.std = torch.tensor(std_init, device=device, dtype=dtype)  # 예: [0.3, 0.6, 0.9]
            

    def sample(self, B):
        return torch.randn(B, self.horizon, self.control_dim, device=self.device, dtype=self.dtype) * self.std
