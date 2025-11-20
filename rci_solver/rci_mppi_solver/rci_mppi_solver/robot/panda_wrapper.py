import os
import numpy as np
from numpy import nan_to_num, sum
from numpy.linalg import pinv

import pinocchio as pin
from pinocchio import RobotWrapper
from pinocchio.utils import *
import jax
import jax.numpy as jnp

class State():
    def __init__(self):
        self.q: np.array
        self.v: np.array
        self.a: np.array

        
        self.q_des: np.array
        self.v_des: np.array
        self.a_des: np.array
        self.q_ref: np.array
        self.v_ref: np.array

        self.acc: np.array
        self.tau: np.array
        self.torque: np.array
        self.v_input: np.array

        self.nq: np.array
        self.nv: np.array
        self.na: np.array

        self.id: np.array
        self.G: np.array
        self.M: np.array
        self.J: np.array
        self.M_q: np.array
        self.oMi : pin.SE3
        self.oMi_offset : pin.SE3

        self.q_jnp: jnp.ndarray
        self.v_jnp: jnp.ndarray
        self.a_jnp: jnp.ndarray

        self.finger_q: np.array
        self.finger_v: np.array
        self.finger_q_jnp: jnp.ndarray
        self.finger_v_jnp: jnp.ndarray

class PandaWrapper(RobotWrapper):
    def __init__(self, model_path,ee_joint_name):
        self.model_path = model_path
        self.__models = self.BuildFromURDF(model_path)
        self.data, self.__collision_data, self.__visual_data = \
            pin.createDatas(self.__models.model, self.__models.collision_model, self.__models.visual_model)
        self.model = self.__models.model
        self.state = State()

        self.ee_joint_name = ee_joint_name
        self.state.id = self.index(self.ee_joint_name)

        self.state.nq = self.model.nq
        self.state.nv = self.model.nv
        self.state.na = self.model.nv

        self.state.q = zero(self.state.nq)
        self.state.v = zero(self.state.nv)
        self.state.a = zero(self.state.na)
        self.state.acc = zero(self.state.na)
        self.state.tau = zero(self.state.nv)

        self.state.oMi = pin.SE3()
        self.state.oMi_offest = pin.SE3()

         # === Offset Transformation ===
        theta_z = -np.pi / 4
        theta_x = np.pi
        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z),  np.cos(theta_z), 0],
            [0,                0,               1]
        ])

        Rx = np.array([
            [1, 0,              0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x),  np.cos(theta_x)]
        ])

        R_offset = Rz @ Rx
        t_offset = np.array([0.0, 0.0, 0.2])
        self.T_offset = pin.SE3(R_offset, t_offset)
        

    def computeAllTerms(self):
        pin.computeAllTerms(self.model, self.data, self.state.q, self.state.v)
        # pin.compute
        self.state.G = self.nle(self.state.q, self.state.v)     # NonLinearEffects
        self.state.M = self.mass(self.state.q)                  # Mass
        self.state.J = self.getJointJacobian(self.state.id)
        self.state.a = pin.aba(self.model, self.data, self.state.q, self.state.v, self.state.tau)
        self.state.oMi = self.data.oMi[self.state.id]
        self.state.oMi_offset = self.state.oMi * self.T_offset
        # print("state.q",self.state.q)
        # print("state.v",self.state.v)


