import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from core.registry import SAMPLERS
from interfaces.sampler import BaseNoiseSampler


@SAMPLERS.register("uniform")
class UniformSampler(BaseNoiseSampler):
    def __init__(self, control_dim, horizon, std_init=0.5, device="cpu", dtype=torch.float32):
        super().__init__(control_dim, horizon)
        self.device = device
        self.dtype = dtype

        # std_init: float or list or tensor
        if isinstance(std_init, (float, int)):
            self.std = torch.full((control_dim,), float(std_init), device=device, dtype=dtype)
        else:
            self.std = torch.as_tensor(std_init, device=device, dtype=dtype)
            if self.std.numel() == 1:
                self.std = self.std.expand(control_dim)
            assert self.std.shape == (control_dim,), "std_init shape mismatch"

        # Uniform 분포의 half-range a = sqrt(3)*σ 로 변환
        # self.range = (3.0 ** 0.5) * self.std
        self.range = 6.0

    def sample(self, B):
        # U[-a, a] 샘플링
        noise = (torch.rand(B, self.horizon, self.control_dim, device=self.device, dtype=self.dtype) * 2 - 1)
        return noise * self.range
