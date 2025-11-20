from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class DynamicsSpec:
    state_dim: int
    control_dim: int
    dt: float

class BaseDynamics(ABC):
    def __init__(self, spec: DynamicsSpec, device="cpu", dtype=None):
        self.spec = spec
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def f(self, x, u):
        """물리 수식: x_{t+1} = f(x,u)"""
        ...

    def step(self, x, u):
        """통일된 호출 규약"""
        return self.f(x, u)
