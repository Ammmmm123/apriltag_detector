"""
pose_estimator.py
-----------------
位姿解算工具类，封装 PnP 求解、旋转矩阵互转等数学方法。
与 ROS2 完全解耦，方便单元测试。
"""

import math
import numpy as np
import cv2


class PoseEstimator:
    """
    利用相机内参对 AprilTag 进行 PnP 位姿解算。

    Parameters
    ----------
    camera_matrix : np.ndarray, shape (3, 3)
        相机内参矩阵 K。
    dist_coeffs : np.ndarray, shape (1, 5) 或 (5,)
        畸变系数 [k1, k2, p1, p2, k3]。
    """

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.camera_matrix = camera_matrix.astype(np.float64)
        self.dist_coeffs = dist_coeffs.astype(np.float64).flatten()

    # ------------------------------------------------------------------
    # 核心求解
    # ------------------------------------------------------------------

    def solve_pnp(
        self, corners: np.ndarray, tag_size: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        由 AprilTag 四角点（图像像素坐标）解算位姿。

        参数
        ----
        corners : np.ndarray, shape (4, 2)
            标签四个角点的图像坐标，顺序为左上、右上、右下、左下
            （pupil-apriltags 返回格式）。
        tag_size : float
            标签实际边长，单位：米。

        返回
        ----
        rvec : np.ndarray, shape (3, 1)
            旋转向量（Rodrigues）。
        tvec : np.ndarray, shape (3, 1)
            平移向量，单位：米，表示标签中心相对摄像头的位置。
        """
        half = tag_size / 2.0
        # 标签在自身坐标系下的四角点（右手系，z=0 平面）
        # 顺序对应 pupil-apriltags：左上、右上、右下、左下
        obj_pts = np.array(
            [
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )
        img_pts = corners.reshape(4, 1, 2).astype(np.float64)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            raise RuntimeError("cv2.solvePnP 求解失败")
        return rvec, tvec

    # ------------------------------------------------------------------
    # 旋转转换
    # ------------------------------------------------------------------

    def rvec_to_rotation_matrix(self, rvec: np.ndarray) -> np.ndarray:
        """旋转向量 → 3×3 旋转矩阵。"""
        R, _ = cv2.Rodrigues(rvec)
        return R

    def rvec_to_euler(
        self, rvec: np.ndarray
    ) -> tuple[float, float, float]:
        """
        旋转向量 → 欧拉角 (roll, pitch, yaw)，单位：弧度。

        采用 ZYX（yaw-pitch-roll）内旋顺序，与航空/机器人惯例一致。
        """
        R = self.rvec_to_rotation_matrix(rvec)
        # ZYX 内旋：R = Rz(yaw) * Ry(pitch) * Rx(roll)
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll  = math.atan2( R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = math.atan2( R[1, 0], R[0, 0])
        else:
            roll  = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = 0.0
        return roll, pitch, yaw

    def rvec_to_quaternion(
        self, rvec: np.ndarray
    ) -> tuple[float, float, float, float]:
        """
        旋转向量 → 四元数 (x, y, z, w)，符合 ROS2 geometry_msgs 标准。
        """
        R = self.rvec_to_rotation_matrix(rvec)
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return x, y, z, w

    # ------------------------------------------------------------------
    # 距离计算
    # ------------------------------------------------------------------

    @staticmethod
    def translation_to_distance(tvec: np.ndarray) -> float:
        """
        平移向量 → 欧氏距离（标签中心到摄像头光心），单位：米。
        """
        return float(np.linalg.norm(tvec))

    # ------------------------------------------------------------------
    # 调试辅助
    # ------------------------------------------------------------------

    def draw_axis(
        self,
        frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        axis_length: float = 0.05,
    ) -> np.ndarray:
        """
        在图像上绘制三轴坐标（红=X，绿=Y，蓝=Z）。
        """
        cv2.drawFrameAxes(
            frame,
            self.camera_matrix,
            self.dist_coeffs,
            rvec,
            tvec,
            axis_length,
        )
        return frame
