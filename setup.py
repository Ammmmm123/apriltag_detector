from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'apriltag_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 安装 launch 文件
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # 安装配置文件
        (os.path.join('share', package_name, 'config'),
            glob('apriltag_detector/config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='ROS2 Python package for USB camera AprilTag detection',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 摄像头采集节点
            'camera_node = apriltag_detector.camera_node:main',
            # AprilTag 检测节点
            'apriltag_detector_node = apriltag_detector.apriltag_detector_node:main',
        ],
    },
)
