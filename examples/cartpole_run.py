# examples/cartpole_run.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import numpy as np
from __init__ import build_controller
from defaults.mppi import MPPIConfig
from utils.cartpole_renderer import render_cartpole_gif
import time
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--theta0", type=float, default=3.14)  # 초기 각도(rad)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--lambda_", type=float, default=1.0)
    p.add_argument("--u_clip", type=float, default=10.0)
    p.add_argument("--std_init", type=float, default=5.0)
    p.add_argument("--save", type=str, default="outputs/cartpole.gif")
    args = p.parse_args()

    # 1) 컨트롤러 구성
    cfg = MPPIConfig(
        horizon=args.horizon,
        samples=args.samples,
        lambda_=args.lambda_,
        gamma=1.0,
        u_clip=args.u_clip,
        device=args.device,
        dtype=torch.float32,
    )

    ctrl = build_controller(
        cfg,
        dynamics_name="cartpole",
        cost_name="quadratic",
        sampler_name="gaussian",
        dynamics_cfg={"dt": args.dt, "angle_wrap": True, "device": args.device},
        cost_cfg={"Q": [3.0, 1.0, 5.0, 0.5], "R": [0.01], "x_goal": (0., 0., 0., 0.), "device": args.device},
        sampler_cfg={"std_init": args.std_init, "device": args.device},
    )

    # 2) 초기 상태: [x, x_dot, theta, theta_dot]
    x = torch.tensor([[0.0, 0.0, args.theta0, 0.0]], device=args.device)

    # 3) 로그 버퍼
    T = args.steps
    ts = np.arange(T, dtype=float) * args.dt
    xs = np.zeros((T, 4), dtype=float)
    us = np.zeros((T,), dtype=float)

    # 4) 실행 루프
    for t in range(T):
        stime = time.perf_counter()
        u = ctrl.step(x)                          # [U] torch tensor
        x = ctrl.f.step(x, u.unsqueeze(0))        # 다음 상태
        print("time: ", time.perf_counter()-stime)
        xs[t, :] = x.squeeze(0).detach().cpu().numpy()
        us[t] = float(u.item())

    print("[done] final state:", xs[-1].tolist())

    # 5) 애니메이션 렌더
    out_path = render_cartpole_gif(ts, xs, pole_half_length=0.5, save_path=args.save)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()