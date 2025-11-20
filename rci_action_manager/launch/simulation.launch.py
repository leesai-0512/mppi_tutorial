import os
import launch
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():

    mujoco_yaml_path = os.path.join(
        get_package_share_directory('rci_action_manager'),
        'config',
        'mujoco_params.yaml'
    )
    simulation_yaml_path = os.path.join(
        get_package_share_directory('rci_action_manager'),
        'config',
        'simulation_params.yaml'
    )
    

    return launch.LaunchDescription([

        Node(
            package='mujoco_ros2',
            executable='mujoco_ros2_node',
            name='mujoco_ros2_node',
            output='screen',
            parameters=[mujoco_yaml_path],
        ),
        Node(
            package='rci_action_manager',
            executable='simulation_node',
            name='simulation_node',
            output='screen',
            parameters=[simulation_yaml_path],
        ),
        

    ])
