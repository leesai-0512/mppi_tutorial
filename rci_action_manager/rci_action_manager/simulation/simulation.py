import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import numpy as np
import pinocchio as pin
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R

from rci_mppi_solver.robot.panda_wrapper import PandaWrapper
from rci_mppi_solver.util.util import Util
from rci_action_manager.controller.controller import RobotController

import time
import jax
import jax.numpy as jnp

class Simulation(Node):
    def __init__(self):
        super().__init__('simulation_node')

        self.declare_parameter('description_package_name', '')
        self.declare_parameter('urdf_file_path', '')
        self.declare_parameter('ee_joint_name', '')
        self.declare_parameter('joint_names', [''])
        
        
        description_package_name = self.get_parameter('description_package_name').get_parameter_value().string_value
        urdf_file_path = self.get_parameter('urdf_file_path').get_parameter_value().string_value
        model_path = os.path.join(get_package_share_directory(description_package_name), urdf_file_path)
        ee_joint_name = self.get_parameter('ee_joint_name').get_parameter_value().string_value
        self.joint_names = self.get_parameter("joint_names").get_parameter_value().string_array_value

        self.joint_state_ready = False
        self.object_state_ready = False

        self.set_target = True
        self.reach = True
        self.grasp = False
        self.move = False
        self.ungrasp = False

        device = jax.devices()[0]
        print(device)  # ex: GPU or CPU

        self.robot = PandaWrapper(model_path,ee_joint_name)
        self.ctrl = RobotController(self.robot, self.get_clock())

        
        # 상태 변수 초기화
        self.q = np.zeros(self.robot.state.nq)
        self.v = np.zeros(self.robot.state.nv)
        self.G = np.zeros(self.robot.state.nv)

        self.joint_state_sub = self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
        self.object_state_sub = self.create_subscription(PoseStamped, "/object_pose", self.object_state_callback,10)
        self.torque_pub = self.create_publisher(JointState, "/effort_controller/commands", 10)
        self.grasp_pub = self.create_publisher(Bool, "/grasp_flag",10)

        self.get_logger().info("Simulation Node Initialized")
        self.stime = time.perf_counter()
        # 루프 타이머 설정 (100Hz)
        self.create_timer(0.01, self.control_loop)

    def joint_state_callback(self, msg):
        
        self.q = np.array(msg.position[:self.robot.state.nq])
        self.v = np.array(msg.velocity[:self.robot.state.nv])
        self.finger_q = np.array(msg.position[self.robot.state.nq:self.robot.state.nq + 2])
        self.finger_v = np.array(msg.velocity[self.robot.state.nv:self.robot.state.nv + 2])
        self.joint_state_ready = True
        self.robot.state.q = self.q.copy()
        self.robot.state.v = self.v.copy()
        self.robot.state.finger_q = self.finger_q.copy()
        self.robot.state.finger_v = self.finger_v.copy()

        self.robot.computeAllTerms()

        self.robot.state.q_jnp = jnp.array(self.q)
        self.robot.state.v_jnp = jnp.array(self.v)
        self.robot.state.finger_q_jnp = jnp.array(self.finger_q)
        self.robot.state.finger_v_jnp = jnp.array(self.finger_v)
        # print("oMi: ", self.robot.state.oMi.translation)
        
    def object_state_callback(self,msg):
        target_pos = msg.pose.position
        target_ori = msg.pose.orientation

        pos = jnp.array([target_pos.x, target_pos.y, target_pos.z])
        quat_np = np.array([target_ori.x, target_ori.y, target_ori.z, target_ori.w])
        rot_mat_np = R.from_quat(quat_np).as_matrix()
        rot_mat = jnp.array(rot_mat_np)


        T = jnp.eye(4)
        T = T.at[:3, :3].set(rot_mat)
        T = T.at[:3, 3].set(pos)

        if self.set_target:
            self.ctrl.goal = T
            # self.ctrl.goal = self.ctrl.goal.at[0, 3].set(self.robot.state.oMi.translation[0] )
            # self.ctrl.goal = self.ctrl.goal.at[1, 3].set(self.robot.state.oMi.translation[1] + 0.2)
            # self.ctrl.goal = self.ctrl.goal.at[2, 3].set(self.robot.state.oMi.translation[2] - 0.2)
            self.set_target = False
        # print("goal: ", self.ctrl.goal[2,3])
        self.object_state_ready = True

    def ok(self):
        return self.joint_state_ready and self.object_state_ready

    def control_loop(self):
        # self.get_logger().info(f"oMi : {self.robot.state.oMi}")
        if self.ok():
            # self.se3Server.compute()
            self.ctrl.controlSe3()
            if self.ctrl.controlFlag:
                des_u = self.ctrl.des_u.copy()
            else:
                des_u = self.robot.state.G.copy()

            torque_msg = JointState()
            torque_msg.name = self.joint_names
            torque_msg.effort = des_u.tolist()
            self.torque_pub.publish(torque_msg)

            if not Util.is_grasp(self.robot.state.finger_q,self.robot.state.finger_v) and Util.goal_checker_only_pos(self.ctrl.goal, self.robot.state.oMi_offset) and self.reach:
                print("goal arrive and grasp",flush=True)
                ftime = time.perf_counter()
                print("time: ", ftime - self.stime,flush=True)
                grasp_msg = Bool()
                grasp_msg.data = True
                self.grasp_pub.publish(grasp_msg)
                self.grasp = True
                self.reach = False

            if Util.is_grasp(self.robot.state.finger_q,self.robot.state.finger_v) and self.grasp:
                print("up object")
                # self.ctrl.goal[0,3] += 0.3
                self.ctrl.goal = self.ctrl.goal.at[2, 3].add(0.3)
                # self.ctrl.mppiSolver.se3_goal_cost.w_rot = 0.0
                # self.ctrl.mppiSolver.se3_goal_cost.w_rot_terminal = 0.0
                self.grasp = False
                self.move = True

            if Util.is_grasp(self.robot.state.finger_q,self.robot.state.finger_v) and Util.goal_checker_only_pos(self.ctrl.goal, self.robot.state.oMi_offset, pos_tol=6e-2)and self.move:
                print("move object")
                self.ctrl.goal = self.ctrl.goal.at[1, 3].add(0.3)
                # self.ctrl.mppiSolver.se3_goal_cost.w_rot = 0.0
                # self.ctrl.mppiSolver.se3_goal_cost.w_rot_terminal = 0.0
                self.move = False
                self.ungrasp = True
                

            if Util.is_grasp(self.robot.state.finger_q,self.robot.state.finger_v) and Util.goal_checker_only_pos(self.ctrl.goal, self.robot.state.oMi_offset,pos_tol=1e-1) and self.ungrasp:
                print("goal arrive and ungrasp")
                grasp_msg = Bool()
                grasp_msg.data = False
                self.grasp_pub.publish(grasp_msg)
                self.ungrasp = False
    

            # self.get_logger().info(f"Published gravity torque: {self.G}")

def main():
    rclpy.init()
    node = Simulation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
