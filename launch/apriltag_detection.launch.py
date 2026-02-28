from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('apriltag_detector')
    default_camera_params = os.path.join(pkg_share, 'config', 'camera_params.yaml')

    return LaunchDescription([
        # ── 可调参数 ──────────────────────────────────────────────────
        DeclareLaunchArgument('device_id',   default_value='0'),
        DeclareLaunchArgument('width',       default_value='1280'),
        DeclareLaunchArgument('height',      default_value='720'),
        DeclareLaunchArgument('fps',          default_value='30'),
        # focus_absolute 范围：0~1023（值越大=越近，值越小=越远/无穷远）
        # 超焦距约 2.3m，对准 2~3m 处由 AF 读取的实测值
        # TODO: 将 -1 替换为实测超焦距值（对准 2~3m 处，AF 收敛后读取）
        # 命令：v4l2-ctl -d /dev/video0 --get-ctrl=focus_absolute
        DeclareLaunchArgument('focus_absolute', default_value='580'),
        DeclareLaunchArgument('focus_auto',     default_value='false'),
        DeclareLaunchArgument('tag_family',  default_value='tag36h11'),
        DeclareLaunchArgument('tag_size',    default_value='0.133'),
        DeclareLaunchArgument('target_tag_id', default_value='4'),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        DeclareLaunchArgument('camera_params_file',  default_value=default_camera_params),

        # ── 摄像头采集节点 ────────────────────────────────────────────
        Node(
            package='apriltag_detector',
            executable='camera_node',
            name='usb_camera_node',
            parameters=[{
                'device_id':          LaunchConfiguration('device_id'),
                'width':              LaunchConfiguration('width'),
                'height':             LaunchConfiguration('height'),
                'fps':                LaunchConfiguration('fps'),
                'use_mjpg':           True,
                'focus_absolute':     LaunchConfiguration('focus_absolute'),
                'focus_auto':         LaunchConfiguration('focus_auto'),
                'camera_params_file': LaunchConfiguration('camera_params_file'),
            }],
            output='screen',
        ),

        # ── AprilTag 检测节点 ─────────────────────────────────────────
        Node(
            package='apriltag_detector',
            executable='apriltag_detector_node',
            name='apriltag_detector_node',
            parameters=[{
                'tag_family':         LaunchConfiguration('tag_family'),
                'tag_size':           LaunchConfiguration('tag_size'),
                'target_tag_id':      LaunchConfiguration('target_tag_id'),
                'publish_debug_image': LaunchConfiguration('publish_debug_image'),
                'camera_params_file': LaunchConfiguration('camera_params_file'),
            }],
            output='screen',
        ),
    ])
