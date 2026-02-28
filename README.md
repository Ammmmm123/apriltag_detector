# apriltag_detector

ROS2 Humble Python 功能包，用于通过 USB 摄像头实时检测 AprilTag，并发布标签相对摄像头的 **三维位置** 和 **姿态/角度** 信息。

---

## 摄像头规格（AT-01M5400B-V1）

| 参数 | 值 |
|------|----|
| 型号 | AT-01M5400B-V1 |
| 传感器 | 1200 万像素 CMOS，1/2.55" |
| 像元尺寸 | 1.4 μm × 1.4 μm |
| 对焦方式 | PDAF 快速自动对焦 |
| 接口 | USB 2.0，UVC 免驱 |
| 图像格式 | MJPG / YUY2 |
| 推荐分辨率 | 1920×1080 @ 30 fps (MJPG) |
| 工作电压/电流 | USB-DC5V / 610 mA |

> **注意**：相机内参（fx, fy, cx, cy, 畸变系数）需通过 `camera_calibration` 标定工具实际测量后填入 `config/camera_params.yaml`。
> 以 1920×1080 分辨率、像元尺寸 1.4 μm 为参考，若镜头焦距约为 3.6 mm，则理论 fx ≈ fy ≈ 3.6 / 0.0014 ≈ **2571 px**（仅供参考，以实际标定结果为准）。

---

## 依赖的 Python 包

### ROS2 / 系统包（apt 安装）

```bash
sudo apt install \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    python3-opencv
```

### Python 第三方包（pip 安装）

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `pupil-apriltags` | ≥ 1.0.4 | AprilTag 检测核心库（基于 apriltag3 C 库的 Python 绑定） |
| `numpy` | ≥ 1.21 | 矩阵运算、位姿变换 |
| `opencv-python` | ≥ 4.5 | 图像采集、预处理、畸变校正 |
| `scipy` | ≥ 1.7 | 旋转矩阵 ↔ 欧拉角/四元数转换（可选） |
| `transforms3d` | ≥ 0.4 | 四元数/旋转矩阵互转（可选，也可用 scipy） |

```bash
pip install pupil-apriltags numpy scipy transforms3d
```

> `opencv-python` 建议使用系统 apt 版本（`python3-opencv`）以避免与 ROS2 冲突。

---

## 功能包结构

```
apriltag_detector/
├── README.md
├── package.xml
├── setup.cfg
├── setup.py
├── resource/
│   └── apriltag_detector           # ROS2 ament 资源标记文件
├── launch/
│   └── apriltag_detection.launch.py  # 一键启动 launch 文件
├── apriltag_detector/
│   ├── __init__.py
│   ├── camera_node.py              # 摄像头采集节点
│   ├── apriltag_detector_node.py   # AprilTag 检测与位姿发布节点
│   ├── pose_estimator.py           # 位姿解算工具类
│   └── config/
│       └── camera_params.yaml      # 相机内参与畸变系数配置
```

---

## 节点说明

### 1. `UsbCameraNode`（`camera_node.py`）

**功能**：通过 OpenCV 打开 USB UVC 摄像头，以指定分辨率和帧率采集图像，发布 ROS2 图像话题。

**类结构**：
```python
class UsbCameraNode(rclpy.node.Node):
    # 构造函数：读取参数（设备号、分辨率、帧率、格式）、初始化 cv2.VideoCapture
    def __init__(self)

    # 定时器回调：读取一帧，转换为 sensor_msgs/Image，发布
    def _timer_callback(self) -> None

    # 设置 V4L2 / OpenCV 摄像头属性（分辨率、MJPG 格式等）
    def _configure_camera(self) -> None

    # 节点关闭时释放摄像头资源
    def destroy_node(self) -> None
```

**发布话题**：

| 话题名 | 消息类型 | 说明 |
|--------|---------|------|
| `/camera/image_raw` | `sensor_msgs/msg/Image` | 原始图像（BGR） |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | 相机内参信息 |

**ROS2 参数**：

| 参数名 | 默认值 | 说明 |
|--------|-------|------|
| `device_id` | `0` | `/dev/video0` 对应设备号 |
| `width` | `1920` | 图像宽度（像素） |
| `height` | `1080` | 图像高度（像素） |
| `fps` | `30` | 帧率 |
| `use_mjpg` | `True` | 是否强制使用 MJPG 格式 |
| `camera_params_file` | `config/camera_params.yaml` | 内参文件路径 |

---

### 2. `AprilTagDetectorNode`（`apriltag_detector_node.py`）

**功能**：订阅图像话题，使用 `pupil-apriltags` 检测图像中的 AprilTag，计算每个标签相对于摄像头的 **3D 位置**（x, y, z）和 **姿态**（四元数 + 欧拉角），将结果发布到相应话题。

**类结构**：
```python
class AprilTagDetectorNode(rclpy.node.Node):
    # 构造函数：读取参数、初始化 Detector、创建订阅/发布器
    def __init__(self)

    # 图像订阅回调：cv_bridge 转换 → 灰度化 → 检测 → 位姿解算 → 发布
    def _image_callback(self, msg: Image) -> None

    # 将 pupil-apriltags 的检测结果转换为 ROS2 PoseStamped 消息列表
    def _detections_to_poses(self, detections: list) -> list[PoseStamped]

    # 在图像上绘制检测框、坐标轴、ID、距离等调试信息（可选，发布到 /camera/image_debug）
    def _draw_debug(self, frame: np.ndarray, detections: list) -> np.ndarray

    # 加载相机内参文件，返回 (camera_matrix, dist_coeffs)
    def _load_camera_params(self, path: str) -> tuple
```

**订阅话题**：

| 话题名 | 消息类型 | 说明 |
|--------|---------|------|
| `/camera/image_raw` | `sensor_msgs/msg/Image` | 原始图像 |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | 相机内参（可选，优先于文件） |

**发布话题**：

| 话题名 | 消息类型 | 说明 |
|--------|---------|------|
| `/apriltag/detections` | `geometry_msgs/msg/PoseArray` | 所有检测到的标签位姿数组 |
| `/apriltag/pose` | `geometry_msgs/msg/PoseStamped` | 主标签（ID=0 或最近标签）位姿 |
| `/apriltag/relative_position` | `geometry_msgs/msg/Vector3Stamped` | 标签相对摄像头位置 (x, y, z)，单位：米 |
| `/apriltag/euler_angles` | `geometry_msgs/msg/Vector3Stamped` | 标签相对摄像头欧拉角 (roll, pitch, yaw)，单位：弧度 |
| `/apriltag/distance` | `std_msgs/msg/Float64` | 标签到摄像头中心的直线距离，单位：米 |
| `/camera/image_debug` | `sensor_msgs/msg/Image` | 带检测框叠加的调试图像 |

**ROS2 参数**：

| 参数名 | 默认值 | 说明 |
|--------|-------|------|
| `tag_family` | `tag36h11` | AprilTag 家族（tag36h11 / tag25h9 / tag16h5） |
| `tag_size` | `0.15` | 实际标签边长，单位：米（必须准确测量） |
| `max_hamming` | `0` | 最大允许汉明距离（0=最严格） |
| `target_tag_id` | `-1` | 目标追踪 ID（-1=追踪最近标签） |
| `publish_debug_image` | `True` | 是否发布带叠加信息的调试图像 |
| `camera_params_file` | `config/camera_params.yaml` | 内参文件路径 |

---

### 3. `PoseEstimator`（`pose_estimator.py`）

**功能**：工具类，封装位姿解算的数学方法，与 ROS2 解耦，方便单元测试。

**类结构**：
```python
class PoseEstimator:
    # 构造函数：接受相机内参矩阵和畸变系数
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray)

    # 由 AprilTag 四角点解算 PnP，返回旋转向量和平移向量
    def solve_pnp(self, corners: np.ndarray, tag_size: float) -> tuple[np.ndarray, np.ndarray]

    # 旋转向量 → 旋转矩阵 → 欧拉角 (roll, pitch, yaw)
    def rvec_to_euler(self, rvec: np.ndarray) -> tuple[float, float, float]

    # 旋转向量 → 四元数 (x, y, z, w)
    def rvec_to_quaternion(self, rvec: np.ndarray) -> tuple[float, float, float, float]

    # 计算标签到摄像头的直线距离
    @staticmethod
    def translation_to_distance(tvec: np.ndarray) -> float
```

---

## 坐标系约定

```
摄像头坐标系（右手系，光轴为 Z 轴正方向）：
  x → 图像右方
  y → 图像下方
  z → 光轴（远离摄像头方向）

发布的 relative_position:
  x = 水平偏移（正值：标签在摄像头右侧）
  y = 垂直偏移（正值：标签在摄像头下方）
  z = 纵深距离（正值：标签在摄像头前方）

发布的 euler_angles (roll, pitch, yaw):
  以摄像头坐标系为参考，描述标签平面的旋转
```

---

## 快速开始

### 1. 安装依赖

```bash
sudo apt install ros-humble-cv-bridge ros-humble-image-transport \
                 ros-humble-tf2-ros ros-humble-tf2-geometry-msgs python3-opencv
pip install pupil-apriltags numpy scipy
```

### 2. 标定相机（推荐）

```bash
ros2 run camera_calibration cameracalibrator \
    --size 8x6 --square 0.025 \
    image:=/camera/image_raw camera:=/camera
```

将标定结果填入 `apriltag_detector/config/camera_params.yaml`。

### 3. 编译并运行

```bash
cd ~/ws_mpcland
colcon build --packages-select apriltag_detector
source install/setup.bash

# 方式一：launch 文件启动
ros2 launch apriltag_detector apriltag_detection.launch.py

# 方式二：分开启动（调试用）
ros2 run apriltag_detector camera_node
ros2 run apriltag_detector apriltag_detector_node
```

### 4. 查看结果

```bash
# 查看位置信息
ros2 topic echo /apriltag/relative_position

# 查看欧拉角
ros2 topic echo /apriltag/euler_angles

# 查看距离
ros2 topic echo /apriltag/distance

# 查看调试图像
ros2 run rqt_image_view rqt_image_view /camera/image_debug
```

---

## config/camera_params.yaml 格式

```yaml
# 相机内参（需通过标定获得，以下为 1920x1080 理论估计值，仅供参考）
camera_matrix:
  rows: 3
  cols: 3
  data: [2571.0, 0.0, 960.0,
          0.0, 2571.0, 540.0,
          0.0,    0.0,   1.0]

# 畸变系数 [k1, k2, p1, p2, k3]
dist_coeffs:
  rows: 1
  cols: 5
  data: [0.0, 0.0, 0.0, 0.0, 0.0]

# 图像尺寸
image_width: 1920
image_height: 1080
```

---

## 功能实现状态

- [x] `UsbCameraNode`：USB 摄像头采集与发布
- [x] `PoseEstimator`：位姿解算工具类
- [x] `AprilTagDetectorNode`：AprilTag 检测与位姿发布
- [x] `camera_params.yaml`：默认相机内参配置文件
- [x] `apriltag_detection.launch.py`：一键启动 launch 文件
- [x] `package.xml` / `setup.py`：ROS2 包元信息
- [ ] 支持多标签同时追踪（当前已发布 PoseArray，可扩展）
- [ ] 发布 TF 变换（`/tf`）
- [ ] 支持通过 ROS2 参数动态调整检测参数
