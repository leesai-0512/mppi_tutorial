# examples/quadrotor3d_run.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch

from __init__ import build_controller
from defaults.mppi import MPPIConfig
from utils.quadrotor3d_renderer import render_quad3d_gif


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--dt", type=float, default=0.02)

    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--samples", type=int, default=50000)
    p.add_argument("--lambda_", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.99)

    # u = [T, tau_phi, tau_theta, tau_psi]
    p.add_argument("--u_clip", type=float, nargs='+', default=[15.0, 0.5, 0.5, 0.5])
    p.add_argument("--std_init", type=float, nargs='+', default=[2.0, 0.3, 0.3, 0.1])

    # 초기/목표 상태 (위치 + yaw만 신경, 나머지는 0에서 시작)
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--y0", type=float, default=0.0)
    p.add_argument("--z0", type=float, default=1.0)

    p.add_argument("--gx", type=float, default=0.0)
    p.add_argument("--gy", type=float, default=0.0)
    p.add_argument("--gz", type=float, default=2.0)

    p.add_argument("--record_sample", type=bool, default=False)
    p.add_argument("--save", type=str, default="outputs/quad3d.gif")
    args = p.parse_args()

    # 1) MPPI 컨피그
    cfg = MPPIConfig(
        horizon=args.horizon,
        samples=args.samples,
        lambda_=args.lambda_,
        gamma=args.gamma,
        u_clip=args.u_clip,          # [T, tau_phi, tau_theta, tau_psi]
        device=args.device,
        dtype=torch.float32,
        record_sample=args.record_sample,
    )

    # 2) 컨트롤러 구성
    # 상태: [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
    x_goal = (
        args.gx, args.gy, args.gz,   # 위치
        0.0, 0.0, 0.0,               # 속도
        0.0, 0.0, 0.0,               # roll, pitch, yaw
        0.0, 0.0, 0.0,               # 각속도
    )

    ctrl = build_controller(
        cfg,
        dynamics_name="quadrotor3d",
        cost_name="quadratic",
        sampler_name="gaussian",
        dynamics_cfg={
            "dt": args.dt,
            "mass": 1.0,
            "Jx": 0.02,
            "Jy": 0.02,
            "Jz": 0.04,
            "thrust_min": 0.0,
            "thrust_max": 20.0,
            "torque_clip": 0.5,
            "angle_wrap": True,
            # device/dtype는 build_controller가 넘김
        },
        cost_cfg={
            # 상태 12차원에 대한 대각 Q
            # [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
            "Q": [8.0, 8.0, 10.0,   # 위치
                  1.0, 1.0, 1.0,    # 속도
                  2.0, 2.0, 1.0,    # roll,pitch,yaw
                  0.5, 0.5, 0.5],   # 각속도
            "R": [0.01, 0.01, 0.01, 0.01],  # 입력 페널티
            "x_goal": x_goal,
            "device": args.device,
        },
        sampler_cfg={
            "std_init": args.std_init,
            "device": args.device,
        },
    )

    # 3) 초기 상태: [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
    x0 = torch.tensor(
        [[args.x0, args.y0, args.z0,
          0.0, 0.0, 0.0,
          0.0, 0.0, 0.0,
          0.0, 0.0, 0.0]],
        device=args.device
    )
    x = x0

    # 4) 로그 버퍼
    T = args.steps
    ts = np.arange(T, dtype=float) * args.dt

    # 궤적은 위치만 저장해서 렌더러로 넘김 [px,py,pz]
    xs = np.zeros((T, 3), dtype=float)
    us = np.zeros((T, 4), dtype=float)

    if args.record_sample:
        # Xss[t] : [samples, horizon, 3] (= [B, T, xyz])
        Xss = np.zeros((T, args.samples, args.horizon, 3), dtype=float)
    else:
        Xss = None

    # 5) 실행 루프
    for t in range(T):
        if args.record_sample:
            # ctrl.step이 (u, Xs, noise)를 리턴한다고 가정
            u, Xs, noise = ctrl.step(x)     # u:[4], Xs:[B, T+1, 12]
            # 계획 horizon 구간의 위치만 저장 (pre-action 기준: Xs[:,1:,:3])
            Xss[t, ...] = Xs[:, 1:, :3].detach().cpu().numpy()

            if t == 0:
                os.makedirs("outputs", exist_ok=True)
                np.save("outputs/quad_sample_noise_gaussian.npy",
                        noise[:, 0, :].detach().cpu().numpy())
                np.save("outputs/quad_sample_traj_gaussian.npy",
                        Xs[:, :, :3].detach().cpu().numpy())
        else:
            u = ctrl.step(x)

        # 실제 시스템 다음 상태 전파
        x = ctrl.f.step(x, u.unsqueeze(0))      # [1,12]
        px, py, pz = x[0, 0].item(), x[0, 1].item(), x[0, 2].item()

        xs[t, :] = np.array([px, py, pz], dtype=float)
        us[t, :] = u.detach().cpu().numpy()

        if (t % 20) == 0:
            print(f"t={t:03d} pos=({px:+.2f},{py:+.2f},{pz:+.2f}) "
                  f"u=[{us[t,0]:+.2f},{us[t,1]:+.2f},{us[t,2]:+.2f},{us[t,3]:+.2f}]")

    print("[done] final state (pos):", xs[-1].tolist())

    # 6) 애니메이션 렌더 (3D)
    out_path = render_quad3d_gif(
        ts, xs, Xss,
        save_path=args.save,
        goal=(args.gx, args.gy, args.gz),
    )
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
