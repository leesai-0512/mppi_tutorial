from rci_mppi_solver.util.util import Util
import jax.numpy as jnp

class CostSE3Ori:
    def __init__(self):

        self.w_ori = 100.0
        self.w_ori_terminal = 1000.0
        self.util = Util()

    def compute(self,T):

        R_ee = T[..., :3, :3] # B,T,3,3

        q_ee = Util.rotmat_to_quat(R_ee)

        q_ee = q_ee / (jnp.linalg.norm(q_ee, axis=-1, keepdims=True) + 1e-8)

        roll_pitch_sq = jnp.square(q_ee[..., 1]) + jnp.square(q_ee[..., 2])
        running_ori_cost = jnp.sum(roll_pitch_sq[:, :-1], axis=-1)
        terminal_ori_cost = roll_pitch_sq[:, -1] 
        cost = self.w_ori * running_ori_cost + self.w_ori_terminal * terminal_ori_cost


        return cost
        