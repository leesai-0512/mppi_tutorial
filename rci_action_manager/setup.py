from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rci_action_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools','rci_mppi_solver'],
    zip_safe=True,
    maintainer='home',
    maintainer_email='home@todo.todo',
    description='RCI Action Manager for manipulator  control',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simulation_node = rci_action_manager.simulation.simulation:main',
            'rci_client_node = rci_action_manager.client.rci_client:main',

        ],
    },
)
