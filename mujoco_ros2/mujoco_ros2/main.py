import os
import rclpy
from rclpy.node import Node
from .mujoco_manager import MujocoManager
from .robot_interface.franka_handler import FrankaHandler
import mujoco
import mujoco.viewer
import time
from ament_index_python.packages import get_package_share_directory

class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_ros2_node')

        self.declare_parameter('description_package_name', '')
        self.declare_parameter('xml_file_path', '')
        self.declare_parameter('initial_qpos',[0.0])
        description_package_name = self.get_parameter('description_package_name').get_parameter_value().string_value
        xml_file_path = self.get_parameter('xml_file_path').get_parameter_value().string_value
        model_path = os.path.join(get_package_share_directory(description_package_name), xml_file_path)
        init_qpos = self.get_parameter('initial_qpos').get_parameter_value().double_array_value

        
        self.manager = MujocoManager(model_path)
        self.model = self.manager.get_model()
        self.data = self.manager.get_data()
        joint_names = self.manager.get_joint_names()
        self.robot = FrankaHandler(self, self.model, self.data, base_index=0,
                                           robot_joint_names=joint_names, init_pose=init_qpos)

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while rclpy.ok() and viewer.is_running():
                
                self.robot.publish_joint_state()
                self.robot.publish_target_pose()
                if not self.robot.ok():
                    rclpy.spin_once(self, timeout_sec=0.01)
                    continue
                # self.robot.publish_ft_sensor()
                self.robot.apply_commands()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                rclpy.spin_once(self, timeout_sec=0.01)

def main():
    rclpy.init()
    node = MujocoSimNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
