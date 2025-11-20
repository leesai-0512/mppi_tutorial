import numpy as np
import math
from rci_mppi_solver.costs.cost_se3_goal import CostSE3Goal
from rci_mppi_solver.costs.cost_control_input import CostControlInput
from rci_mppi_solver.costs.cost_velocity_zero import CostVelZero     
from rci_mppi_solver.costs.cost_se3_ori import CostSE3Ori  

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from functools import partial
class RCIMPPISolver:
    def __init__(
            self,
            batch_size,
            time_step,
            dt,
            std,
            goal,
            state,
            lamda,
            ):
        self.batch_size = batch_size
        self.time_step = time_step
        self.dt = dt
        self.robot_state = state
        self.nq = state.nq
        self.nv = state.nv
        self.na = state.nv
        self.std = std
        self.var_inv = 1.0 / (self.std ** 2)
        self.cov = jnp.diag(self.std**2)
        self.cov_inv = jnp.diag(1.0/(self.std**2))
        self.goal = goal
        self.lamda = lamda

        self.dhparams = jnp.array([
            [ 0.0000, 0.3330,     0.0      ],       # Joint 1
            [ 0.0000, 0.0000, -math.pi/2   ],       # Joint 2
            [ 0.0000, 0.3160,  math.pi/2   ],       # Joint 3
            [ 0.0825, 0.0000,  math.pi/2   ],       # Joint 4
            [-0.0825, 0.3840, -math.pi/2   ],       # Joint 5
            [ 0.0000, 0.0000,  math.pi/2   ],       # Joint 6
            [ 0.0880, 0.0000,  math.pi/2   ],       # Joint 7
        ], dtype=jnp.float32)
        
        self.S = jnp.zeros((self.batch_size,), dtype=jnp.float32)

        self.noise = jnp.zeros((self.batch_size, self.time_step, self.na), dtype=jnp.float32)
        self.u_mean = jnp.zeros((self.time_step, self.na), dtype=jnp.float32)

        self.eeTraj = jnp.zeros((self.batch_size, self.time_step, 6), dtype=jnp.float32)
        self.qTraj = jnp.zeros((self.batch_size, self.time_step, self.nq), dtype=jnp.float32)
        self.vTraj = jnp.zeros((self.batch_size, self.time_step, self.nv), dtype=jnp.float32)
        self.T = jnp.zeros((self.batch_size, self.time_step, self.nq, 4, 4), dtype=jnp.float32)

        # offset
        theta = jnp.array(-math.pi / 4, dtype=jnp.float32)
        Rz = jnp.array([
            [jnp.cos(theta), -jnp.sin(theta), 0.0],
            [jnp.sin(theta),  jnp.cos(theta), 0.0],
            [0.0,             0.0,            1.0]
        ], dtype=jnp.float32)

        # Rx: X축 회전 (pi)
        theta = jnp.array(math.pi, dtype=jnp.float32)
        Rx = jnp.array([
            [1.0, 0.0,              0.0],
            [0.0, jnp.cos(theta), -jnp.sin(theta)],
            [0.0, jnp.sin(theta),  jnp.cos(theta)]
        ], dtype=jnp.float32)

        R_offset = Rz @ Rx
        T_offset = jnp.eye(4, dtype=jnp.float32)
        T_offset = T_offset.at[:3, :3].set(R_offset)
        T_offset = T_offset.at[2, 3].set(0.2)
        self.T_offset = jnp.broadcast_to(
            T_offset.reshape(1, 1, 4, 4),
            (self.batch_size, self.time_step, 4, 4)
        )
        
        #cost
        self.se3_goal_cost = CostSE3Goal()
        self.control_input_cost = CostControlInput()
        self.velocity_zero_cost = CostVelZero()
        self.se3_ori_cost = CostSE3Ori()

    @partial(jax.jit, static_argnames=["batch_size", "time_step", "na"])
    def sampling_noise(self, rng_key, u_mean):
        noise = jax.random.normal(rng_key, shape=(self.batch_size, self.time_step, self.na)) * self.std
        u = u_mean + noise
        return noise, u
            
    @jax.jit
    def computeJointStateTraj(self, u):
        da = u * self.dt 
        vTraj = self.robot_state.v_jnp[:, None, :] + jnp.cumsum(da, axis=1)

        dv = vTraj * self.dt
        qTraj = self.robot_state.q_jnp[:, None, :] + jnp.cumsum(dv, axis=1)

        return qTraj, vTraj

    
    @jax.jit
    def computeFKbatch(self, qTraj):
        a = self.dhparams[:, 0].reshape(1, 1, self.nq)
        d = self.dhparams[:, 1].reshape(1, 1, self.nq)
        alpha = self.dhparams[:, 2].reshape(1, 1, self.nq)
        theta = qTraj

        ct = jnp.cos(theta)
        st = jnp.sin(theta)
        ca = jnp.cos(alpha)
        sa = jnp.sin(alpha)

        T = jnp.zeros((self.batch_size, self.time_step, self.nq, 4, 4), dtype=jnp.float32)
        T = T.at[..., 0, 0].set(ct)
        T = T.at[..., 0, 1].set(-st)
        T = T.at[..., 0, 2].set(0.0)
        T = T.at[..., 0, 3].set(a)

        T = T.at[..., 1, 0].set(st * ca)
        T = T.at[..., 1, 1].set(ct * ca)
        T = T.at[..., 1, 2].set(-sa)
        T = T.at[..., 1, 3].set(-sa * d)

        T = T.at[..., 2, 0].set(st * sa)
        T = T.at[..., 2, 1].set(ct * sa)
        T = T.at[..., 2, 2].set(ca)
        T = T.at[..., 2, 3].set(ca * d)

        T = T.at[..., 3, 3].set(1.0)

        T_perm = jnp.transpose(T, (2, 0, 1, 3, 4))
        T07 = jnp.einsum(
            'btij,btjk,btkl,btlm,btmn,btno,btop->btip',
            T_perm[0], T_perm[1], T_perm[2], T_perm[3],
            T_perm[4], T_perm[5], T_perm[6]
        )
        
        T0G = jnp.matmul(T07, self.T_offset)
        return T0G
    
    @jax.jit
    def computeCost(self):
        T = self.computeFKbatch()
        se3_goal_cost = self.se3_goal_cost.compute(T, self.goal)
        control_cost = self.control_input_cost.compute(self.noise,self.var_inv)
        velzero_cost = self.velocity_zero_cost.compute(self.vTraj)
        se3_ori_cost = self.se3_ori_cost.compute(T)
        S = se3_goal_cost + control_cost + velzero_cost + se3_ori_cost
        return S
    
    def computeWeight(self,S):
        S = S - jnp.min(S)
        w = jax.nn.softmax(-S / self.lamda, axis=0)
        return w.reshape(-1, 1, 1)
    
    def savgol_filter_jax(seq, window_size, polyorder):
        assert window_size % 2 == 1, "window_size must be odd"
        assert polyorder < window_size, "polyorder must be < window_size"

        if seq.ndim == 2:
            seq = seq[jnp.newaxis, ...]  # (T, C) → (1, T, C)

        B, T, C = seq.shape

        # (1) SG 커널 계산
        half = window_size // 2
        x = jnp.arange(-half, half + 1, dtype=seq.dtype)  # (-w//2 ... w//2)
        A = jnp.vander(x, N=polyorder + 1, increasing=True)  # (w, p+1)
        ATA = A.T @ A
        AT = A.T
        ATA_inv = jnp.linalg.inv(ATA)
        coeff = ATA_inv @ AT
        coeff = coeff[0][::-1]  # zero-th derivative + reverse → shape (w,)

        # (2) 커널 shape: (C, 1, k) for depthwise conv1d
        kernel = coeff.reshape(1, 1, -1)
        kernel = jnp.tile(kernel, (C, 1, 1))  # (C,1,k)

        # (3) padding
        seq = jnp.transpose(seq, (0, 2, 1))  # (B,C,T)
        pad_width = [(0, 0), (0, 0), (half, half)]  # reflect pad on time dim
        seq_padded = jnp.pad(seq, pad_width, mode='reflect')

        # (4) depthwise conv1d via lax
        def depthwise_conv1d(x, kernel):
            return jax.lax.conv_general_dilated(
                lhs=x, rhs=kernel,
                window_strides=(1,),
                padding="VALID",
                dimension_numbers=('NCH', 'OIH', 'NCH'),
                feature_group_count=C
            )

        out = depthwise_conv1d(seq_padded, kernel)  # (B,C,T)
        out = jnp.transpose(out, (0, 2, 1))  # (B,T,C)
        return out[0] if out.shape[0] == 1 else out
        

    def setGoal(self,goal):
        self.goal = goal

    def solveMPPI(self, rng_key):
        # 1. 샘플링 + smoothing
        noise, u = self.sampling_noise(rng_key, self.u_mean)
        noise = self.savgol_filter_jax(noise, window_size=self.time_step - 1, polyorder=3)
        noise = noise.at[-1, :, :].set(0.0)
        u = noise + self.u_mean
        u = u.at[-2, :, :].set(0.0)

        # 2. Joint State Traj
        qTraj, vTraj = self.computeJointStateTraj(u)

        # 3. FK, Cost
        T = self.computeFKbatch(qTraj)
        S = self.computeCost(T, noise, vTraj)

        # 4. Weighting
        w = self.compute_weight(S, self.lamda)  # -> (B,1,1)
        w_epsilon = jnp.sum(w * noise, axis=0)  # (T, na)
        u_star = self.u_mean + w_epsilon

        # 5. Update mean
        u_mean_new = u_star[1:, :]
        u_mean_new = jnp.concatenate([u_mean_new, jnp.zeros((1, self.na))], axis=0)
        u_mean_new = self.savgol_filter_jax(u_mean_new, window_size=self.time_step - 1, polyorder=3)

        # 6. 저장
        self.u_mean = u_mean_new
        self.noise = noise
        self.u = u

        return jax.device_get(u_star[0])
