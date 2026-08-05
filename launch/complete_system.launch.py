from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


GZ_SETUP = 'source ~/ws/install/setup.bash && '


def generate_launch_description():
    return LaunchDescription([
        # ── 1. 相机内参桥接 ───────────────────────────────────────────
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                GZ_SETUP + 'exec ros2 run ros_gz_bridge parameter_bridge '
                '/world/apritag_car_x500/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/camera_info'
                '@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            ],
            name='bridge_camera_info',
            output='screen',
        ),

        # ── 2. 键盘遥控小车 ───────────────────────────────────────────
        Node(
            package='teleop_twist_keyboard',
            executable='teleop_twist_keyboard',
            name='teleop_twist_keyboard',
            output='screen',
            # teleop_twist_keyboard 需要终端交互，启动后请点击该终端窗口以控制小车
            prefix='x-terminal-emulator -e',
        ),

        # ── 3. 图像数据桥接 ──────────────────────────────────────────
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                GZ_SETUP + 'exec ros2 run ros_gz_bridge parameter_bridge '
                '/world/apritag_car_x500/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image'
                '@sensor_msgs/msg/Image[gz.msgs.Image',
            ],
            name='bridge_camera_image',
            output='screen',
        ),

        # ── 4. 小车遥控指令桥接 ──────────────────────────────────────
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                GZ_SETUP + 'exec ros2 run ros_gz_bridge parameter_bridge '
                '/model/apritag_car/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            ],
            name='bridge_cmd_vel',
            output='screen',
        ),

        # ── 5. Topic 转发：/cmd_vel → /model/apritag_car/cmd_vel ────
        Node(
            package='topic_tools',
            executable='relay',
            name='relay_cmd_vel',
            arguments=['/cmd_vel', '/model/apritag_car/cmd_vel'],
            output='screen',
        ),

        # ── 6. AprilTag 码检测节点 ────────────────────────────────────
        Node(
            package='apriltag_detector',
            executable='apriltag_detector_node',
            name='apriltag_detector_node',
            output='screen',
        ),
    ])
