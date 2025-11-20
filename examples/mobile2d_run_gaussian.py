# examples/2dmobile_run.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch

from __init__ import build_controller
from defaults.mppi import MPPIConfig
from utils.mobile2d_renderer import render_mobile2d_gif


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--dt", type=float, default=0.02)

    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--lambda_", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)

    p.add_argument("--u_clip", type=float, nargs='+', default=[0.7, 3.0])
    p.add_argument("--std_init", type=float, nargs='+', default=[0.1, 2.0])

    # 초기/목표 상태
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--y0", type=float, default=0.0)
    p.add_argument("--th0", type=float, default=1.5708)
    p.add_argument("--gx", type=float, default=1.0)
    p.add_argument("--gy", type=float, default=1.0)
    p.add_argument("--gth", type=float, default=1.5708)

    p.add_argument("--record_sample", type=bool, default=True)
    p.add_argument("--save", type=str, default="outputs/mobile2d_gaussian.gif")
    args = p.parse_args()

    # 1) 컨트롤러 구성
    cfg = MPPIConfig(
        horizon=args.horizon,
        samples=args.samples,
        lambda_=args.lambda_,
        gamma=args.gamma,
        u_clip=args.u_clip,              # v, omega는 dynamics에서 각각 클립
        device=args.device,
        dtype=torch.float32,
        record_sample=args.record_sample,
    )

    ctrl = build_controller(
        cfg,
        dynamics_name="mobile2d",
        cost_name="quadratic",
        sampler_name="gaussian",
        dynamics_cfg={
            "dt": args.dt,
            "angle_wrap": True,
            # device/dtype는 build_controller가 이미 넘겨줌 (중복 금지)
        },
        cost_cfg={
            # 상태: [x, y, theta], 입력: [v, omega]
            "Q": [5.0, 5.0, 0.0],   # 위치 우선, heading은 느슨하게
            "R": [0.0, 0.0],      # 제어 페널티
            "x_goal": (args.gx, args.gy, args.gth),
            "device": args.device
        },
        sampler_cfg={
            "std_init": args.std_init, "device": args.device
        },
    )
    
    # 2) 초기 상태: [x, y, theta]
    x = torch.tensor([[args.x0, args.y0, args.th0]], device=args.device)

    # 3) 로그 버퍼
    T = args.steps
    ts = np.arange(T, dtype=float) * args.dt
    xs = np.zeros((T, 3), dtype=float)
    us = np.zeros((T, 2), dtype=float)

    if args.record_sample:
        Xss = np.zeros((T, args.samples, args.horizon, 2), dtype=float)
    else:
        Xss = None
    # 4) 실행 루프
    for t in range(T):
        if args.record_sample:
            u, Xs, noise = ctrl.step(x)                             # [2]
            Xss[t,...] = Xs[:,1:,:2].detach().cpu().numpy()
            if t == 0:
                np.save("outputs/sample_noise_gaussian.npy", noise[:,0,:].detach().cpu().numpy())
                np.save("outputs/sample_traj_gaussian.npy", Xs[:,:,:2].detach().cpu().numpy())
        else:
            u = ctrl.step(x)

        x = ctrl.f.step(x, u.unsqueeze(0))          # [1,3]
        xs[t, :] = x.squeeze(0).detach().cpu().numpy()
        us[t, :] = u.detach().cpu().numpy()

        if (t % 20) == 0:
            px, py, th = xs[t]
            print(f"t={t:03d}  pos=({px:+.2f},{py:+.2f})  th={th:+.2f}  u=[{us[t,0]:+.2f},{us[t,1]:+.2f}]")

    print("[done] final state:", xs[-1].tolist())

    # 5) 애니메이션 렌더
    out_path = render_mobile2d_gif(
        ts, xs, Xss,
        save_path=args.save,
        body_radius=0.15,
        goal=(args.gx, args.gy),
    )
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
