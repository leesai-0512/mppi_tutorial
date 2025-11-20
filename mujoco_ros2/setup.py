from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mujoco_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(where='.'),  # ✅ 하위 모든 폴더 인식
    package_dir={'': '.'},              # ✅ 루트 기준으로 import 경로 설정
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='home',
    maintainer_email='home@todo.todo',
    description='MuJoCo ROS 2 Simulation',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mujoco_ros2_node = mujoco_ros2.main:main',  # ✅ main.py의 main 함수
        ],
    },
)
