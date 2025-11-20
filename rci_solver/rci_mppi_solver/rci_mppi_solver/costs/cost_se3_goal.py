from rci_mppi_solver.util.util import Util
import jax.numpy as jnp

class CostSE3Goal:
    def __init__(self):
        self.w_pos = 100.0
        self.w_rot = 100.0
        self.w_pos_terminal = 1000.0
        self.w_rot_terminal = 1000.0
        self.util = Util()

    def compute(self,T, goal):
        pos_ee = T[..., :3, 3] # B,T,3
        pos_goal = goal[:3, 3]  # 3,
        pos_diff = pos_ee - pos_goal # B,T,3
        pos_cost = jnp.sum(pos_diff ** 2, axis=-1) # B.T



        R_ee = T[..., :3, :3] # B,T,3,3
        R_goal = jnp.broadcast_to(goal[:3, :3], R_ee.shape)

        q_ee = Util.rotmat_to_quat(R_ee)
        q_goal = Util.rotmat_to_quat(R_goal)

        q_ee = q_ee / (jnp.linalg.norm(q_ee, axis=-1, keepdims=True) + 1e-8)
        q_goal = q_goal / (jnp.linalg.norm(q_goal, axis=-1, keepdims=True) + 1e-8)
        
        dot = jnp.sum(q_ee * q_goal, axis=-1)
        rot_cost = (1.0 - dot**2) # B,T

        running_pos_cost = jnp.sum(pos_cost[:, :-1], axis=-1) * self.w_pos
        running_rot_cost = jnp.sum(rot_cost[:, :-1], axis=-1) * self.w_rot

        terminal_pos_cost = pos_cost[:, -1] * self.w_pos_terminal
        terminal_rot_cost = rot_cost[:, -1] * self.w_rot_terminal

        cost = running_pos_cost + running_rot_cost + terminal_pos_cost + terminal_rot_cost  # (B,)
        return cost
        