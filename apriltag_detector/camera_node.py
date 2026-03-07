"""
camera_node.py
--------------
UsbCameraNode：通过 OpenCV 采集 USB UVC 摄像头图像，
发布 /camera/image_raw（sensor_msgs/Image）和
     /camera/camera_info（sensor_msgs/CameraInfo）。

摄像头型号：AT-01M5400B-V1
支持格式：MJPG / YUY2，UVC 免驱。
"""

import os
import subprocess
import threading
import yaml

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory


class UsbCameraNode(Node):
    """
    通过 OpenCV 打开 USB UVC 摄像头并发布图像话题。

    ROS2 参数
    ---------
    device_id          : int   摄像头设备号（默认 0 → /dev/video0）
    width              : int   图像宽度像素（默认 1280）
    height             : int   图像高度像素（默认 720）
    fps                : int   目标帧率（默认 30）
    use_mjpg           : bool  是否强制使用 MJPG 格式（默认 True）
    camera_params_file : str   相机内参 YAML 路径
    frame_id           : str   发布帧 ID（默认 "camera_optical_frame"）
    """

    def __init__(self):
        super().__init__('usb_camera_node')

        # ── 声明参数 ───────────────────────────────────────────────────
        self.declare_parameter('device_id', 1)
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 30)
        self.declare_parameter('use_mjpg', True)
        self.declare_parameter('frame_id', 'camera_optical_frame')
        # 对焦锁定参数：-1 表示不干预（保持自动对焦）
        self.declare_parameter('focus_absolute', 580)
        self.declare_parameter('focus_auto', True)
        # 曝光参数：exposure_auto=False 关闭自动曝光，exposure_absolute 设定固定曝光时间（单位：100μs）
        # 自动曝光在低光/变光场景下会把曝光时间拉到 1~3s，导致 cap.read() 随机阻塞
        # 建议：先开自动曝光让相机收敛，读取 exposure_absolute 值，再锁定
        # 命令：v4l2-ctl -d /dev/video0 --get-ctrl=exposure_absolute
        self.declare_parameter('exposure_auto', True)    # True=自动曝光，False=手动
        self.declare_parameter('exposure_absolute', 300) # 单位100μs，300=30ms≈33fps上限

        default_cfg = os.path.join(
            get_package_share_directory('apriltag_detector'),
            'config', 'camera_params.yaml'
        )
        self.declare_parameter('camera_params_file', default_cfg)

        # ── 读取参数 ───────────────────────────────────────────────────
        self._device_id  = self.get_parameter('device_id').value
        self._width      = self.get_parameter('width').value
        self._height     = self.get_parameter('height').value
        self._fps        = self.get_parameter('fps').value
        self._use_mjpg      = self.get_parameter('use_mjpg').value
        self._frame_id      = self.get_parameter('frame_id').value
        self._cfg_path      = self.get_parameter('camera_params_file').value
        self._focus_abs     = self.get_parameter('focus_absolute').value
        self._focus_auto    = self.get_parameter('focus_auto').value
        self._exposure_auto = self.get_parameter('exposure_auto').value
        self._exposure_abs  = self.get_parameter('exposure_absolute').value

        # ── 相机内参 ───────────────────────────────────────────────────
        self._camera_info_msg = self._build_camera_info(self._cfg_path)

        # ── QoS ───────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── 发布器 ────────────────────────────────────────────────────
        self._pub_image = self.create_publisher(Image,      '/camera/image_raw',  qos)
        self._pub_info  = self.create_publisher(CameraInfo, '/camera/camera_info', qos)
        self._bridge    = CvBridge()
        # ── 对焦控制（在 VideoCapture 之前，通过 v4l2-ctl 设置）────────
        self._lock_focus()
        self._lock_exposure()
        # ── 打开摄像头 ────────────────────────────────────────────────
        self._cap = None
        self._configure_camera()

        # ── 采集线程（替代定时器，cap.read() 阻塞不影响 ROS 执行器）──────
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name='camera_capture'
        )
        self._capture_thread.start()
        self.get_logger().info(
            f'UsbCameraNode 启动：/dev/video{self._device_id} '
            f'{self._width}×{self._height} @ {self._fps}fps'
        )

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _lock_exposure(self) -> None:
        """
        通过 v4l2-ctl 锁定曝光，防止自动曝光将曝光时间拉长导致 cap.read() 随机阻塞 1~3s。

        参数说明：
          exposure_auto     = True  → 保持自动曝光（默认）
          exposure_auto     = False → 关闭自动曝光，启用手动
          exposure_absolute = N    → 曝光时间，单位 100μs。
                                     例：300 = 30ms，对应最高 33fps；
                                         166 = 16.6ms，对应最高 60fps。
        """
        dev = f'/dev/video{self._device_id}'

        if self._exposure_auto:
            # 确保自动曝光开启（防止上次会话残留手动状态）
            for val in ('3', '1'):   # 3=aperture priority(UVC), 1=manual off
                for ctrl in ('exposure_auto',):
                    ret = subprocess.run(
                        ['v4l2-ctl', '-d', dev, f'--set-ctrl={ctrl}={val}'],
                        capture_output=True, text=True
                    )
                    if ret.returncode == 0:
                        self.get_logger().info(f'已开启自动曝光（{ctrl}={val}）')
                        return
            self.get_logger().warn('未能开启自动曝光，如帧率过低请设置 exposure_auto:=false')
            return

        # 关闭自动曝光（UVC 手动模式值为 1）
        for val in ('1',):
            ret = subprocess.run(
                ['v4l2-ctl', '-d', dev, f'--set-ctrl=exposure_auto={val}'],
                capture_output=True, text=True
            )
            if ret.returncode == 0:
                self.get_logger().info('已关闭自动曝光（手动模式）')
                break
        else:
            self.get_logger().warn('未能关闭自动曝光')
            return

        # 设置固定曝光时间
        ret = subprocess.run(
            ['v4l2-ctl', '-d', dev,
             f'--set-ctrl=exposure_absolute={self._exposure_abs}'],
            capture_output=True, text=True
        )
        if ret.returncode == 0:
            self.get_logger().info(
                f'曝光已锁定：exposure_absolute={self._exposure_abs} '
                f'({self._exposure_abs * 0.1:.1f}ms)'
            )
        else:
            self.get_logger().warn(
                f'设置 exposure_absolute 失败：{ret.stderr.strip()}\n'
                f'请手动执行：v4l2-ctl -d {dev} --set-ctrl=exposure_absolute={self._exposure_abs}'
            )

    def _lock_focus(self) -> None:
        """
        通过 v4l2-ctl 锁定摄像头焦距。

        参数说明：
          focus_auto     = True  → 保持自动对焦（默认，不执行任何操作）
          focus_auto     = False → 关闭自动对焦
          focus_absolute = -1   → 不设定具体值（仅关闭 AF）
          focus_absolute = N    → 锁定到数值 N（需先关闭 AF）
        """
        dev = f'/dev/video{self._device_id}'

        if self._focus_auto:
            # 主动写入寄存器，确保自动对焦真正开启（防止上次会话残留关闭状态）
            for ctrl in ('focus_automatic_continuous', 'auto_focus'):
                ret = subprocess.run(
                    ['v4l2-ctl', '-d', dev, f'--set-ctrl={ctrl}=1'],
                    capture_output=True, text=True
                )
                if ret.returncode == 0:
                    self.get_logger().info(f'已通过 {ctrl}=1 开启自动对焦')
                    break
            else:
                self.get_logger().warn(
                    '未能开启自动对焦，请手动执行：\n'
                    f'  v4l2-ctl -d {dev} --set-ctrl=focus_automatic_continuous=1'
                )
            return

        # 关闭自动对焦
        for ctrl in ('focus_automatic_continuous', 'auto_focus'):
            ret = subprocess.run(
                ['v4l2-ctl', '-d', dev, f'--set-ctrl={ctrl}=0'],
                capture_output=True, text=True
            )
            if ret.returncode == 0:
                self.get_logger().info(f'已通过 {ctrl}=0 关闭自动对焦')
                break
        else:
            self.get_logger().warn(
                '未能关闭自动对焦，请手动执行：\n'
                f'  v4l2-ctl -d {dev} --set-ctrl=focus_automatic_continuous=0'
            )

        # 设置固定焦距值
        if self._focus_abs >= 0:
            ret = subprocess.run(
                ['v4l2-ctl', '-d', dev,
                 f'--set-ctrl=focus_absolute={self._focus_abs}'],
                capture_output=True, text=True
            )
            if ret.returncode == 0:
                self.get_logger().info(
                    f'焦距已锁定：focus_absolute={self._focus_abs}'
                )
            else:
                self.get_logger().warn(
                    f'设置 focus_absolute 失败：{ret.stderr.strip()}'
                )
        else:
            self.get_logger().info(
                '已关闭自动对焦，未设定具体焦距值（使用当前位置）。\n'
                '建议：先自动对焦到目标距离后读取焦距值，\n'
                f'  v4l2-ctl -d {dev} --get-ctrl=focus_absolute\n'
                '再通过参数 focus_absolute:=<值> 固定。'
            )

    def _configure_camera(self) -> None:
        """打开并配置 USB UVC 摄像头。"""
        dev_path = f'/dev/video{self._device_id}'

        # 在 VideoCapture 打开之前，用 v4l2-ctl 预设像素格式。
        # OpenCV 的 cap.set(CAP_PROP_FOURCC) 在部分驱动/版本下对已打开的设备无效，
        # 而驱动在 open() 时就会锁定格式，因此必须在此之前设置。
        # 茄子相机流畅而 OpenCV 卡顿，正是因为 OpenCV 默认以 YUYV 打开，
        # 1280×720 YUYV 每帧 ~1.8MB，USB 2.0 带宽不足以支撑 30fps，
        # 使用 MJPG 可将带宽降低 90%+，恢复 30fps。
        if self._use_mjpg:
            ret = subprocess.run(
                ['v4l2-ctl', '-d', dev_path,
                 '--set-fmt-video',
                 f'width={self._width},height={self._height},pixelformat=MJPG'],
                capture_output=True, text=True
            )
            if ret.returncode == 0:
                self.get_logger().info(
                    f'v4l2-ctl 已预设格式：MJPG {self._width}×{self._height}'
                )
            else:
                self.get_logger().warn(
                    f'v4l2-ctl 预设 MJPG 失败（{ret.stderr.strip()}），'
                    f'将依赖 OpenCV 设置，可能回退到 YUYV 导致帧率低'
                )

        cap = cv2.VideoCapture(self._device_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.get_logger().error(f'无法打开摄像头 {dev_path}')
            raise RuntimeError(f'无法打开摄像头 {dev_path}')

        # 即使已用 v4l2-ctl 预设，也再次 set 作为兜底（对支持的驱动生效）
        if self._use_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS,          self._fps)
        # 将 V4L2 内核缓冲区减到 1，cap.read() 始终取最新帧而不是积压的旧帧
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 读取并打印实际生效的参数
        actual_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps  = cap.get(cv2.CAP_PROP_FPS)
        fourcc_int  = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str  = ''.join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
        self.get_logger().info(
            f'实际分辨率：{actual_w}×{actual_h} @ {actual_fps:.1f}fps  '
            f'格式：{fourcc_str}'
        )
        if self._use_mjpg and fourcc_str.upper() != 'MJPG':
            self.get_logger().warn(
                f'警告：期望 MJPG 但实际格式为 {fourcc_str}，'
                f'帧率可能受 USB 带宽限制（YUYV @ 1280×720 ≈ 5fps）'
            )
        self._cap = cap

    def _capture_loop(self) -> None:
        """采集线程主循环：以最快速度持续调用 cap.read()，直接发布，不经过定时器。"""
        while rclpy.ok():
            if self._cap is None or not self._cap.isOpened():
                self.get_logger().warn('摄像头未就绪，等待重试...')
                import time; time.sleep(0.5)
                continue

            ret, frame = self._cap.read()   # 阻塞直到摄像头交付一帧
            if not ret or frame is None:
                self.get_logger().warn('读取帧失败')
                continue

            stamp = self.get_clock().now().to_msg()

            img_msg = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp    = stamp
            img_msg.header.frame_id = self._frame_id
            self._pub_image.publish(img_msg)

            self._camera_info_msg.header.stamp    = stamp
            self._camera_info_msg.header.frame_id = self._frame_id
            self._pub_info.publish(self._camera_info_msg)

    def _build_camera_info(self, cfg_path: str) -> CameraInfo:
        """从 YAML 文件构建 CameraInfo 消息。"""
        msg = CameraInfo()
        if not os.path.isfile(cfg_path):
            self.get_logger().warn(
                f'相机内参文件不存在：{cfg_path}，将使用零值占位'
            )
            return msg

        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        msg.width  = int(cfg.get('image_width',  self._width))
        msg.height = int(cfg.get('image_height', self._height))

        K = cfg.get('camera_matrix', {}).get('data', [])
        if len(K) == 9:
            msg.k = [float(v) for v in K]

        D = cfg.get('dist_coeffs', {}).get('data', [])
        if D:
            msg.d = [float(v) for v in D]

        R = cfg.get('rectification_matrix', {}).get('data', [])
        if len(R) == 9:
            msg.r = [float(v) for v in R]

        P = cfg.get('projection_matrix', {}).get('data', [])
        if len(P) == 12:
            msg.p = [float(v) for v in P]

        msg.distortion_model = 'plumb_bob'
        return msg

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        """节点销毁时释放摄像头资源。"""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self.get_logger().info('摄像头资源已释放')
        super().destroy_node()


# ── 入口 ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = UsbCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
