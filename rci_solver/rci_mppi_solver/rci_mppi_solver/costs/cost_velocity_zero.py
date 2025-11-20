import jax.numpy as jnp
from rci_mppi_solver.util.util import Util

class CostVelZero:
    def __init__(self):
        self.w_running = 0.0
        self.w_terminal = 10.0
        self.util = Util()

    def compute(self, vel):
        running_vel = vel[:, :-1, :]   
        terminal_vel = vel[:, -1, :]    

        running_cost = self.w_running * jnp.sum(running_vel ** 2, axis=-1).sum(axis=-1) 
        terminal_cost = self.w_terminal * jnp.sum(terminal_vel ** 2, axis=-1)           

        cost = running_cost + terminal_cost
        return cost 