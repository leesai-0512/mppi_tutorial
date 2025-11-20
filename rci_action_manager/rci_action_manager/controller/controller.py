import numpy as np
from rci_mppi_solver.robot.panda_wrapper import PandaWrapper
# from rci_mppi_solver.solver.mppi_solver import RCIMPPISolver
from rci_mppi_solver.solver.mppi_solver_jax import RCIMPPISolver
# from rci_mppi_solver.solver.mppi_solver_compile import RCIMPPISolver

from rclpy.logging import get_logger

import pinocchio as pin
from pinocchio.utils import *
import jax
import jax.numpy as jnp

from copy import deepcopy
import time

class RobotController:
    def __init__(self, robot : PandaWrapper, clock):
        self.robot = robot
        self.clock = clock

        # Control : Time
        self.stime = None
        self.ctime = None
        self.duration  = None

        # Control : Target
        self.targetJoint = None
        self.targetSE3 = pin.SE3()
        
        # Robot Data
        self.nq = self.robot.state.nq


        self.goal = None
        
        # solver
        self.mppiSolver = RCIMPPISolver(
            batch_size=500,
            time_step=30,
            dt=0.01,
            std=jnp.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0]),
            goal=self.goal,
            state=self.robot.state,
            lamda=0.3
        )


        # State : Result
        self.des_u : np.array

        # isControl
        self.controlFlag = True
        
        # qTmp 
        self.qTmp : np.array

        self.logger = get_logger("Controller")


    def initSE3(self, target : pin.SE3, duration):
        self.controlFlag = True

        self.stime = self.clock.now().nanoseconds/1e9
        self.duration = duration
        self.targetSE3 = target
        self.se3Traj.setStartTime(self.stime)
        self.se3Traj.setDuration(self.duration)
        self.se3Traj.setTargetSample(self.targetSE3)
        self.se3Traj.setInitSample(self.robot.state.oMi)


    def controlSe3(self):
        self.ctime = self.clock.now().nanoseconds/1e9
        # self.se3Traj.setCurrentTime(self.ctime)
        # se3Ref = self.se3Traj.computeNext()

        self.mppiSolver.setGoal(self.goal)
        stime = time.perf_counter()
        qddot = self.mppiSolver.solveMPPI()
        ftime = time.perf_counter()


        torque = self.robot.state.M @ qddot + self.robot.state.G
        # print("torque: ", torque)
        self.des_u = np.copy(torque)

