"""
apriltag_detector_node.py
--------------------------
AprilTagDetectorNode：订阅 /camera/image_raw，
使用 pupil-apriltags 检测 AprilTag，计算每个标签相对摄像头的
三维位置与姿态，并发布到多个话题。

发布话题
--------
/apriltag/detections        geometry_msgs/PoseArray
    所有标签位姿（按检测顺序）

/apriltag/pose              geometry_msgs/PoseStamped
/apriltag/relative_position geometry_msgs/Vector3Stamped
/apriltag/euler_angles      geometry_msgs/Vector3Stamped
/apriltag/distance          std_msgs/Float64
    每个检测到的标签均会向以上话题发布一条消息。
    消息的 header.frame_id = "tag_<N>"(如 tag_254)，
    订阅者通过读取 frame_id 即可判断属于哪个码。

/camera/image_debug         sensor_msgs/Image
    调试叠加图像（所有码均标注 ID/位置/距离/坐标轴）
"""

import os
import yaml
import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import (
    PoseArray, PoseStamped, Pose,
    Vector3Stamped, Quaternion
)
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from .pose_estimator import PoseEstimator

try:
    from pupil_apriltags import Detector
    _APRILTAG_AVAILABLE = True
except ImportError:
    _APRILTAG_AVAILABLE = False


class AprilTagDetectorNode(Node):
    """
    AprilTag 检测与位姿发布节点。

    ROS2 参数
    ---------
    tag_family          : str   标签家族（默认 'tag36h11'）
    tag_size            : float 默认标签边长，单位：米（默认 0.133）
    tag_size_map        : str   按 ID 指定边长，格式 "id:size,id:size"，例如 "4:0.133,7:0.080"
                                未在表中的 ID 使用 tag_size 作为回退值
    max_hamming         : int   最大汉明距离（默认 0）
    publish_debug_image : bool  是否发布调试图像（默认 True）
    camera_params_file  : str   相机内参文件路径
    frame_id            : str   发布帧 ID（默认 "camera_optical_frame"）
    """

    # 调试绘图颜色常量（BGR）
    _COLOR_BBOX   = (0,   255,   0)    # 绿色：检测框
    _COLOR_TARGET = (0,    64, 255)    # 橙色：目标框
    _COLOR_TEXT   = (255, 255, 255)    # 白色：文字
    _COLOR_CENTER = (0,    0,  255)    # 红色：中心点

    def __init__(self):
        super().__init__('apriltag_detector_node')

        if not _APRILTAG_AVAILABLE:
            self.get_logger().error(
                'pupil-apriltags 未安装！请运行：pip install pupil-apriltags'
            )
            raise ImportError('pupil-apriltags not found')

        # ── 声明参数 ────────────────────────────────────────────────────
        self.declare_parameter('tag_family', 'tag36h11')
        self.declare_parameter('tag_size', 0.133)
        self.declare_parameter('tag_size_map', '254:0.175,80:0.022')   # 例如 "4:0.133,7:0.080"
        self.declare_parameter('max_hamming', 0)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('frame_id', 'camera_optical_frame')

        default_cfg = os.path.join(
            get_package_share_directory('apriltag_detector'),
            'config', 'camera_params.yaml'
        )
        self.declare_parameter('camera_params_file', default_cfg)

        # ── 读取参数 ────────────────────────────────────────────────────
        self._tag_family      = self.get_parameter('tag_family').value
        self._tag_size        = self.get_parameter('tag_size').value
        self._max_hamming     = self.get_parameter('max_hamming').value
        self._pub_debug       = self.get_parameter('publish_debug_image').value
        self._frame_id        = self.get_parameter('frame_id').value
        self._cfg_path        = self.get_parameter('camera_params_file').value

        # 解析 tag_size_map："id:size,id:size" → {int: float}
        self._tag_size_dict: dict[int, float] = {}
        raw_map = self.get_parameter('tag_size_map').value.strip()
        if raw_map:
            for entry in raw_map.split(','):
                entry = entry.strip()
                if ':' in entry:
                    id_str, sz_str = entry.split(':', 1)
                    try:
                        self._tag_size_dict[int(id_str.strip())] = \
                            float(sz_str.strip())
                    except ValueError:
                        self.get_logger().warn(
                            f'tag_size_map 解析失败，忽略条目："{entry}"'
                        )
        if self._tag_size_dict:
            self.get_logger().info(
                'tag_size_map：' +
                ', '.join(f'ID {k}→{v} m'
                          for k, v in self._tag_size_dict.items())
            )
        else:
            self.get_logger().info(
                f'tag_size_map 未设置，所有标签使用统一尺寸 {self._tag_size} m'
            )

        # ── 相机内参 ─────────────────────────────────────────────────────
        self._camera_matrix, self._dist_coeffs = self._load_camera_params(
            self._cfg_path
        )
        self._pose_estimator = PoseEstimator(
            self._camera_matrix, self._dist_coeffs
        )
        # pupil-apriltags 需要的内参分量
        fx = float(self._camera_matrix[0, 0])
        fy = float(self._camera_matrix[1, 1])
        cx = float(self._camera_matrix[0, 2])
        cy = float(self._camera_matrix[1, 2])
        self._camera_params = (fx, fy, cx, cy)

        # ── 初始化检测器 ─────────────────────────────────────────────────
        self._detector = Detector(
            families=self._tag_family,
            nthreads=2,
            quad_decimate=1.5,   # 下采样加速，适当降低精度换取速度
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        tracked_ids = list(self._tag_size_dict.keys()) if self._tag_size_dict \
            else ['全部']
        self.get_logger().info(
            f'AprilTag 检测器初始化：family={self._tag_family}  '
            f'默认 tag_size={self._tag_size} m  '
            f'检测目标 ID：{tracked_ids}'
        )

        # ── QoS ─────────────────────────────────────────────────────────
        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )

        # ── 订阅 ─────────────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._sub_image = self.create_subscription(
            Image, '/camera/image_raw', self._image_callback, qos_sub
        )
        self._sub_info = self.create_subscription(
            CameraInfo, '/camera/camera_info', self._info_callback, qos_sub
        )

        # ── 发布器 ───────────────────────────────────────────────────────
        # self._pub_poses    = self.create_publisher(PoseArray,      '/apriltag/detections',        qos_pub)
        
        # 所有检测到的码均向以下话题发布，frame_id="tag_<N>"编码标签 ID
        self._pub_pose     = self.create_publisher(PoseStamped,    '/apriltag/pose',               qos_pub)
        self._pub_position = self.create_publisher(Vector3Stamped, '/apriltag/relative_position',  qos_pub)
        self._pub_euler    = self.create_publisher(Vector3Stamped, '/apriltag/euler_angles',       qos_pub)
        self._pub_distance = self.create_publisher(Float64,        '/apriltag/distance',           qos_pub)
        if self._pub_debug:
            self._pub_debug_img = self.create_publisher(
                Image, '/camera/image_debug', qos_sub
            )

        self.get_logger().info('AprilTagDetectorNode 启动完成')

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------

    def _info_callback(self, msg: CameraInfo) -> None:
        """
        订阅 /camera/camera_info，动态更新内参（优先于文件）。
        收到第一帧有效内参后即销毁该订阅。
        """
        if len(msg.k) == 9 and any(v != 0.0 for v in msg.k):
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            D = np.array(msg.d, dtype=np.float64) if msg.d else np.zeros(5)
            self._camera_matrix = K
            self._dist_coeffs   = D
            self._pose_estimator = PoseEstimator(K, D)
            self._camera_params  = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            self.get_logger().info(
                f'已从 /camera/camera_info 更新内参：'
                f'fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  '
                f'cx={K[0,2]:.1f}  cy={K[1,2]:.1f}'
            )
            # 只需更新一次，销毁订阅节省资源
            self.destroy_subscription(self._sub_info)

    def _image_callback(self, msg: Image) -> None:
        """图像订阅回调：检测 → 解算 → 发布。"""
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败：{e}')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # AprilTag 检测
        # estimate_tag_pose=False：角点检测与 tag_size 无关，内置位姿估计
        # 只支持单一尺寸，多尺寸场景下由后续 solve_pnp 按各码真实尺寸解算。
        detections = self._detector.detect(
            gray,
            estimate_tag_pose=False,
            camera_params=self._camera_params,
            tag_size=self._tag_size,
        )

        stamp = msg.header.stamp

        if not detections:
            if self._pub_debug:
                self._publish_debug_image(frame, [], stamp)
            return

        # 发布所有标签 PoseArray  （按检测顺序，未区分 ID，供需要批量处理的订阅者使用）
        # pose_array = PoseArray()
        # pose_array.header.stamp    = stamp
        # pose_array.header.frame_id = self._frame_id
        # for det in detections:
        #     pose_array.poses.append(
        #         self._detection_to_pose(det)
        #     )
        # self._pub_poses.publish(pose_array)

        # 每个检测到的标签均发布到同一套话题，frame_id 编码 tag ID
        for det in detections:
            self._publish_detection(det, stamp)

        # 调试图像
        if self._pub_debug:
            self._publish_debug_image(frame, detections, stamp)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _get_tag_size(self, tag_id: int) -> float:
        """返回指定 tag_id 对应的实际边长（米）；未配置则用默认值。"""
        return self._tag_size_dict.get(tag_id, self._tag_size)

    def _publish_detection(self, det, stamp) -> None:
        """
        对单个检测结果发布到公共话题。
        header.frame_id = "tag_<N>" 编码标签 ID，订阅者通过读取 frame_id 即可区分不同码。
        """
        tag_size = self._get_tag_size(det.tag_id)
        try:
            rvec, tvec = self._pose_estimator.solve_pnp(det.corners, tag_size)
        except RuntimeError as e:
            self.get_logger().warn(f'PnP 解算失败 ID={det.tag_id}：{e}')
            return

        qx, qy, qz, qw = self._pose_estimator.rvec_to_quaternion(rvec)
        roll, pitch, yaw = self._pose_estimator.rvec_to_euler(rvec)
        tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
        distance = PoseEstimator.translation_to_distance(tvec)

        # frame_id 编码 tag ID，订阅者通过此字段区分不同标签
        tag_frame = f'tag_{det.tag_id}'

        ps = PoseStamped()
        ps.header.stamp    = stamp
        ps.header.frame_id = tag_frame
        ps.pose.position.x = tx
        ps.pose.position.y = ty
        ps.pose.position.z = tz
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self._pub_pose.publish(ps)

        pos_msg = Vector3Stamped()
        pos_msg.header.stamp    = stamp
        pos_msg.header.frame_id = tag_frame
        pos_msg.vector.x = tx
        pos_msg.vector.y = ty
        pos_msg.vector.z = tz
        self._pub_position.publish(pos_msg)

        euler_msg = Vector3Stamped()
        euler_msg.header.stamp    = stamp
        euler_msg.header.frame_id = tag_frame
        euler_msg.vector.x = roll
        euler_msg.vector.y = pitch
        euler_msg.vector.z = yaw
        self._pub_euler.publish(euler_msg)

        dist_msg = Float64()
        dist_msg.data = distance
        self._pub_distance.publish(dist_msg)

        self.get_logger().debug(
            f'ID={det.tag_id}  '
            f'pos=({tx:.3f},{ty:.3f},{tz:.3f}) m  '
            f'dist={distance:.3f} m  '
            f'rpy=({math.degrees(roll):.1f},{math.degrees(pitch):.1f},'
            f'{math.degrees(yaw):.1f})°'
        )

    def _detection_to_pose(self, det) -> Pose:
        """
        将单个检测结果转换为 Pose 消息。
        使用 PnP + 该标签对应的实际尺寸，确保多尺寸标签位姿均准确。
        """
        pose = Pose()
        tag_size = self._get_tag_size(det.tag_id)
        try:
            rvec, tvec = self._pose_estimator.solve_pnp(det.corners, tag_size)
            pose.position.x = float(tvec[0])
            pose.position.y = float(tvec[1])
            pose.position.z = float(tvec[2])
            qx, qy, qz, qw = self._pose_estimator.rvec_to_quaternion(rvec)
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
        except Exception as e:
            self.get_logger().warn(
                f'_detection_to_pose PnP 失败 ID={det.tag_id}：{e}'
            )
        return pose

    # ------------------------------------------------------------------
    # 调试图像
    # ------------------------------------------------------------------

    def _publish_debug_image(self, frame: np.ndarray, detections: list,
                              stamp) -> None:
        """在图像上叠加检测信息并发布到 /camera/image_debug。所有码均显示位置/距离/坐标轴。"""
        debug = frame.copy()
        for det in detections:
            corners = det.corners.astype(int)
            color = self._COLOR_BBOX

            # 绘制边框
            cv2.polylines(debug, [corners], True, color, 2)

            # 中心点
            cx_px = int(det.center[0])
            cy_px = int(det.center[1])
            cv2.circle(debug, (cx_px, cy_px), 5, self._COLOR_CENTER, -1)

            # 标签 ID
            cv2.putText(
                debug, f'ID:{det.tag_id}',
                (corners[0][0], corners[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            # 所有码均尝试解算 PnP 并显示位置/距离
            try:
                _sz = self._get_tag_size(det.tag_id)
                rvec, tvec = self._pose_estimator.solve_pnp(det.corners, _sz)
                tx = float(tvec[0])
                ty = float(tvec[1])
                tz = float(tvec[2])
                dist = float(np.linalg.norm(tvec))
                lines = [
                    f'dist={dist:.3f}m',
                    f'x={tx:.3f} y={ty:.3f}',
                    f'z={tz:.3f}',
                ]
                y0 = cy_px + 20
                for i, line in enumerate(lines):
                    cv2.putText(
                        debug, line,
                        (cx_px - 60, y0 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color, 2
                    )
                # 坐标轴
                self._pose_estimator.draw_axis(debug, rvec, tvec, _sz * 0.5)
            except Exception:
                pass

        # 左上角检测统计
        cv2.putText(
            debug,
            f'AprilTags detected: {len(detections)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        img_msg = self._bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        img_msg.header.stamp    = stamp
        img_msg.header.frame_id = self._frame_id
        self._pub_debug_img.publish(img_msg)

    # ------------------------------------------------------------------
    # 内参加载
    # ------------------------------------------------------------------

    def _load_camera_params(
        self, path: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        从 YAML 文件加载相机内参矩阵和畸变系数。

        返回
        ----
        camera_matrix : np.ndarray (3, 3)
        dist_coeffs   : np.ndarray (5,)
        """
        # 默认值（1280×720，2026-02-28 实测标定结果）
        K = np.array(
            [[1048.5781106716322,    0.0,             668.6117761075526],
             [   0.0,            1051.3019596158053,  361.5752738576768],
             [   0.0,               0.0,                1.0           ]],
            dtype=np.float64
        )
        D = np.array(
            [0.09469271704353091, -0.09162682761377387,
             -0.0007778362651798496, 0.0028121896872062577, 0.0],
            dtype=np.float64
        )

        if not os.path.isfile(path):
            self.get_logger().warn(
                f'内参文件不存在：{path}，使用默认估算值（请进行相机标定！）'
            )
            return K, D

        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)

        k_data = cfg.get('camera_matrix', {}).get('data', [])
        if len(k_data) == 9:
            K = np.array(k_data, dtype=np.float64).reshape(3, 3)

        d_data = cfg.get('dist_coeffs', {}).get('data', [])
        if d_data:
            D = np.array(d_data, dtype=np.float64).flatten()

        self.get_logger().info(
            f'已加载内参：fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  '
            f'cx={K[0,2]:.1f}  cy={K[1,2]:.1f}'
        )
        return K, D


# ── 入口 ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
