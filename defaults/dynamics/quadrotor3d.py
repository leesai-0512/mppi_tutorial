import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from dataclasses import dataclass
from core.registry import DYNAMICS
from interfaces.dynamics import BaseDynamics, DynamicsSpec


@DYNAMICS.register("quadrotor3d")
class Quadrotor3DDynamics(BaseDynamics):
    """
    3D 쿼드로터 단순 모델 (12 상태, 4 입력)

    상태 x: [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
        px,py,pz : 월드 좌표 위치 [m]
        vx,vy,vz : 월드 좌표 속도 [m/s]
        roll, pitch, yaw : ZYX 오일러 각 [rad]
        p,q,r    : body 각속도 [rad/s]

    입력 u: [T, tau_phi, tau_theta, tau_psi]
        T         : 총 추력 (body z축 방향, N)
        tau_phi   : roll 토크 [N·m]
        tau_theta : pitch 토크 [N·m]
        tau_psi   : yaw 토크 [N·m]

    """

    def __init__(
        self,
        dt: float = 0.02,
        mass: float = 1.0,
        Jx: float = 0.02,
        Jy: float = 0.02,
        Jz: float = 0.04,
        g: float = 9.81,
        thrust_min: float = 0.0,
        thrust_max: float = 15.0,
        torque_clip: float | None = None,   # 토크 클리핑 (스칼라, 대칭)
        angle_wrap: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        spec = DynamicsSpec(
            state_dim=12,
            control_dim=4,
            dt=dt,
        )
        super().__init__(spec, device=device, dtype=dtype)

        self.mass = mass
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.g = g
        self.dt = dt

        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.torque_clip = torque_clip
        self.angle_wrap = angle_wrap

        self.J = torch.tensor([Jx, Jy, Jz], device=device, dtype=dtype)  # [3]

    @torch.no_grad()
    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x: [B,12]
        u: [B,4] = [T, tau_phi, tau_theta, tau_psi]
        return x_next: [B,12]
        """
        device, dtype = self.device, self.dtype
        x = x.to(device=device, dtype=dtype)
        u = u.to(device=device, dtype=dtype)

        B = x.shape[0]
        dt = self.dt

        T = u[..., 0:1]               # [B,1]
        tau = u[..., 1:4]             # [B,3]

        # ----- 상태 블럭 분해 -----
        # pos = [px, py, pz], vel = [vx, vy, vz],
        # ang = [roll, pitch, yaw], omega = [p, q, r]
        pos   = x[..., 0:3]   # [B,3]
        vel   = x[..., 3:6]   # [B,3]
        ang   = x[..., 6:9]   # [B,3]
        omega = x[..., 9:12]  # [B,3]

        roll  = ang[..., 0:1]     # [B,1]
        pitch = ang[..., 1:2]     # [B,1]
        yaw   = ang[..., 2:3]     # [B,1]

        p = omega[..., 0:1]       # [B,1]
        q = omega[..., 1:2]       # [B,1]
        r = omega[..., 2:3]       # [B,1]

        # ----- 회전 행렬 R(body->world), ZYX -----
        cr = torch.cos(roll);  sr = torch.sin(roll)
        cp = torch.cos(pitch); sp = torch.sin(pitch)
        cy = torch.cos(yaw);   sy = torch.sin(yaw)

        # R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
        R11 = cy*cp
        R12 = cy*sp*sr - sy*cr
        R13 = cy*sp*cr + sy*sr

        R21 = sy*cp
        R22 = sy*sp*sr + cy*cr
        R23 = sy*sp*cr - cy*sr

        R31 = -sp
        R32 = cp*sr
        R33 = cp*cr

        # body z축 단위 벡터가 world에서 어떻게 보이는지: R * e3
        # e3 = [0,0,1] 이므로, 세 번째 열만 가져오면 됨.
        # R_e3: [B,3]
        R_e3 = torch.cat([R13, R23, R33], dim=-1)

        # ----- 선가속도 -----
        # f_world = (T/m) * R_e3 + g
        f_world = (T / self.mass) * R_e3                # [B,3]

        g_vec = torch.tensor([0.0, 0.0, -self.g], device=device, dtype=dtype)
        g_vec = g_vec.view(1,3)                         # [1,3]
        acc = f_world + g_vec                           # [B,3]

        vel_next = vel + dt * acc                       # [B,3]
        pos_next = pos + dt * vel_next                  # [B,3]

        # ----- 각가속도 -----
        # J * w_dot = tau - w × (J w)
        J = self.J.view(1,3)                            # [1,3]
        Jomega = J * omega                              # [B,3]
        cross = torch.cross(omega, Jomega, dim=-1)      # [B,3]
        omega_dot = (tau - cross) / J                   # [B,3]
        omega_next = omega + dt * omega_dot             # [B,3]

        p_n = omega_next[..., 0:1]
        q_n = omega_next[..., 1:2]
        r_n = omega_next[..., 2:3]

        # ----- 오일러 각 kinematics -----
        # [roll_dot, pitch_dot, yaw_dot]^T = E(roll,pitch) * [p,q,r]^T
        tan_p = torch.tan(pitch)
        sec_p = 1.0 / torch.cos(pitch)

        roll_dot  = p + q*sr*tan_p + r*cr*tan_p
        pitch_dot = q*cr - r*sr
        yaw_dot   = q*sr*sec_p + r*cr*sec_p

        roll_next  = roll  + dt * roll_dot
        pitch_next = pitch + dt * pitch_dot
        yaw_next   = yaw   + dt * yaw_dot

        if self.angle_wrap:
            pi = torch.pi
            roll_next  = (roll_next  + pi) % (2*pi) - pi
            pitch_next = (pitch_next + pi) % (2*pi) - pi
            yaw_next   = (yaw_next   + pi) % (2*pi) - pi

        ang_next = torch.cat([roll_next, pitch_next, yaw_next], dim=-1)   # [B,3]

        # ----- 최종 상태 합치기 -----
        # pos_next, vel_next, ang_next, omega_next 모두 [B,3]
        x_next = torch.cat([pos_next, vel_next, ang_next, omega_next], dim=-1)  # [B,12]
        return x_next

