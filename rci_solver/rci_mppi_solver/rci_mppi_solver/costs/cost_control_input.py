from rci_mppi_solver.util.util import Util
import jax.numpy as jnp
class CostControlInput:
    def __init__(self):
        self.w_running = 0.001
        self.w_terminal = 1.0

        self.util = Util()

    def compute(self, noise, var_inv):
        cost = self.w_running * jnp.sum((noise ** 2) * var_inv, axis=-1).sum(axis=-1)
        return cost
        