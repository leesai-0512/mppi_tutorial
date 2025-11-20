import sys
import os

from defaults import costs
from defaults import dynamics
from defaults import samplers
from defaults import mppi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.registry import ALGOS, DYNAMICS, COSTS, SAMPLERS
# defaults를 import해야 @register 들이 실행되어 등록됨
from defaults.mppi import MPPIConfig

def build_controller(cfg: MPPIConfig,
                     dynamics_name="double_integrator",
                     cost_name="quadratic",
                     sampler_name="gaussian",
                     algo_name="mppi",
                     dynamics_cfg=None, cost_cfg=None, sampler_cfg=None):
    dynamics_cfg = dynamics_cfg or {}
    cost_cfg = cost_cfg or {}
    sampler_cfg = sampler_cfg or {}

    Dyn  = DYNAMICS.get(dynamics_name)
    Cost = COSTS.get(cost_name)
    Samp = SAMPLERS.get(sampler_name)
    Algo = ALGOS.get(algo_name)

    dyn  = Dyn(dtype=cfg.dtype, **dynamics_cfg)
    cost = Cost(dtype=cfg.dtype, **cost_cfg)
    smp  = Samp(dyn.spec.control_dim, cfg.horizon, dtype=cfg.dtype, **sampler_cfg)

    return Algo(dyn, cost, smp, cfg)
