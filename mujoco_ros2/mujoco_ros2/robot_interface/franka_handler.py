from rclpy.node import Node
from rclpy.logging import get_logger
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
import numpy as np
from numpy import nan_to_num, sum
from numpy.linalg import pinv

import mujoco


import os

import pinocchio as pin
from pinocchio import RobotWrapper
from pinocchio.utils import *

from ament_index_python.packages import get_package_share_directory


class FrankaHandler:
    def __init__(self, node: Node, model, data, base_index, robot_joint_names,init_pose):
        self.node = node
        self.model = model
        self.data = data
        self.base_index = base_index

        # chan
        # self.frankaWrapper = pandaWrapper()
        self.logger = get_logger("Mujoco_Robot_Handler")
    

        # 프랑카 조인트 이름만 사용
        self.robot_joints = robot_joint_names

        # 센서 인덱스 및 정보
        # self.force_sensor_index = self.model.sensor(f"franka_force").id * 3
        # self.torque_sensor_index = self.model.sensor(f"franka_torque").id * 3
        # self.node.get_logger().info(f"[Franka] force_sensor_index: {self.force_sensor_index}")
        # self.node.get_logger().info(f"[Franka] torque_sensor_index: {self.torque_sensor_index}")

        # 퍼블리셔 및 서브스크라이버 설정
        self.joint_pub = node.create_publisher(JointState, "/joint_states", 10)
        self.object_pub = node.create_publisher(PoseStamped, "/object_pose", 10)
        self.ft_pub = node.create_publisher(WrenchStamped, "/ft_sensor_state", 10)

        node.create_subscription(JointState, "/effort_controller/commands", self.torque_callback, 10)
        node.create_subscription(Bool, "/grasp_flag", self.grasp_callback, 10)
        self.logger.info(f"init : {init_pose}")
        self.set_initial_pose(init_pose)
        self.velocity_cmd = None
        self.torque_cmd = None
        self.torque_ready = False

    def ok(self):
        return self.torque_ready

    def set_initial_pose(self,init_pose):

        initial_qpos = init_pose
        self.data.qpos[:9] = initial_qpos
        self.data.qvel[:] = np.zeros_like(self.data.qvel)


    def grasp_callback(self, msg: Bool):
        # 손가락 조인트에 force_value를 적용
        force_value = -30.0 if msg.data else 50.0
        # self.node.get_logger().info(f"[Franka] Received grasp flag: {msg.data} -> Applying force {force_value} N")
        for name in self.robot_joints:
            if "finger" in name:
                try:
                    act_id = self.model.actuator(name).id
                    self.data.ctrl[act_id] = force_value
                    # self.node.get_logger().info(f"[Franka] Actuator '{name}' set to {force_value} N")
                except Exception as e:
                    self.node.get_logger().warn(f"[Franka] actuator '{name}' control failed: {e}")

    def torque_callback(self, msg):
        self.torque_cmd = msg
        self.torque_ready = True

    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()

        filtered_joints = [j for j in self.robot_joints if "panda" in j]

        for jname in filtered_joints:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qpos_index = self.model.jnt_qposadr[joint_id]
            msg.position.append(self.data.qpos[qpos_index])
            jid = self.model.joint(jname).id
            msg.velocity.append(self.data.qvel[jid])
            qdd = self.data.qfrc_actuator[jid]
            msg.effort.append(qdd)

        msg.name = filtered_joints
        self.joint_pub.publish(msg)



    def publish_target_pose(self):
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        # 프랑카 조인트만 처리
        for i in range(self.model.njnt):
            jtype = self.model.jnt_type[i]
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)

            if jname.startswith("panda_"):
                continue

            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                qpos_adr = self.model.jnt_qposadr[i]
                qpos = self.data.qpos[qpos_adr : qpos_adr + 7]  # [x y z qw qx qy qz]

                msg = PoseStamped()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.header.frame_id = "world"
                msg.pose.position.x = qpos[0]
                msg.pose.position.y = qpos[1]
                msg.pose.position.z = qpos[2]
                msg.pose.orientation.w = qpos[3]
                msg.pose.orientation.x = qpos[4]
                msg.pose.orientation.y = qpos[5]
                msg.pose.orientation.z = qpos[6]

                self.object_pub.publish(msg)


    def apply_commands(self):
        if self.torque_cmd:
            for i, name in enumerate(self.torque_cmd.name):
                if name in self.robot_joints:
                    try:
                        jid = self.model.joint(name).id
                        act_id = self.model.actuator(name).id
                        self.data.ctrl[act_id] = self.torque_cmd.effort[i]
                    except Exception as e:
                        self.node.get_logger().warn(f"[Franka] actuator '{name}' control failed: {e}")

        else:
            print("torque not receive")


    #-----------------------------------FT sensor----------------------------------------
    def adjoint_transform(self, sensor_pos, com_pos, R_sensor):
        r = sensor_pos - com_pos
        skew_r = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        AdT = np.zeros((6, 6))
        AdT[:3, :3] = R_sensor
        AdT[3:, :3] = R_sensor @ skew_r
        AdT[3:, 3:] = R_sensor
        return AdT
    
    def compute_internal_wrench_for_body_and_children(self, root_body_name, zero_acc=False):
        root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
        sensor_body_id = root_body_id
        nbody = self.model.nbody
        descendants = []

        def get_children(parent_id):
            for i in range(nbody):
                if self.model.body_parentid[i] == parent_id:
                    descendants.append(i)
                    get_children(i)

        descendants.append(root_body_id)
        get_children(root_body_id)

        wrench_total = np.zeros(6)
        sensor_pos = self.data.xpos[sensor_body_id]
        R_sensor = self.data.xmat[sensor_body_id].reshape(3, 3).T

        for body_id in descendants:
            mass = self.model.body_mass[body_id]
            R_body = self.data.xmat[body_id].reshape(3, 3)
            inertia_diag = self.model.body_inertia[body_id]
            I = R_body @ np.diag(inertia_diag) @ R_body.T

            if zero_acc:
                a_lin = np.zeros(3)
                a_ang = np.zeros(3)
                omega = np.zeros(3)
            else:
                a_lin = self.data.cacc[body_id, :3]
                a_ang = self.data.cacc[body_id, 3:]
                omega = self.data.cvel[body_id, 3:]

            f_inertial = mass * a_lin
            f_gravity = mass * np.array(self.model.opt.gravity)
            f_body = f_inertial + f_gravity
            tau_body = I @ a_ang + np.cross(omega, I @ omega)

            wrench_com = np.hstack([f_body, tau_body])
            com_pos = self.data.xpos[body_id] + R_body @ self.model.body_ipos[body_id]
            AdT = self.adjoint_transform(sensor_pos, com_pos, R_sensor)
            wrench_at_sensor = AdT @ wrench_com
            wrench_total += wrench_at_sensor

        return wrench_total[:3], wrench_total[3:]
    
    

    def publish_ft_sensor(self):
        f_static, tau_static = self.compute_internal_wrench_for_body_and_children("panda_link7", zero_acc=True)
        f_measured = self.data.sensordata[self.force_sensor_index : self.force_sensor_index + 3]
        t_measured = self.data.sensordata[self.torque_sensor_index : self.torque_sensor_index + 3]
        f_corrected = f_measured + f_static
        t_corrected = t_measured - tau_static

        msg = WrenchStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "franka_ft_sensor"
        msg.wrench.force.x = f_corrected[0]
        msg.wrench.force.y = f_corrected[1]
        msg.wrench.force.z = f_corrected[2]
        msg.wrench.torque.x = t_corrected[0]
        msg.wrench.torque.y = t_corrected[1]
        msg.wrench.torque.z = t_corrected[2]
        self.ft_pub.publish(msg)

    
