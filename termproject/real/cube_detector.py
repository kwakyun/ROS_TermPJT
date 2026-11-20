"""ROS 2 노드: /top_cam 이미지 → 큐브 base_link (x, y, yaw) → /cube_obs

실행(PC, ROS 2 Humble + venv):  python3 real/cube_detector.py [--config real/config/real.yaml]
발행: /cube_obs  std_msgs/Float64MultiArray  data = [x, y, yaw, valid(0/1), stamp_sec]
      /cube_debug sensor_msgs/Image  검출 결과 그림 (rqt_image_view 로 확인)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision import detect_cube, draw_detection, load_homography  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="real/config/real.yaml")
    args, ros_args = ap.parse_known_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float64MultiArray

    class CubeDetector(Node):
        def __init__(self) -> None:
            super().__init__("cube_detector")
            self.det_cfg = cfg["detector"]
            self.calib = load_homography(self.det_cfg["homography"])
            self.bridge = CvBridge()
            self.pub = self.create_publisher(Float64MultiArray, cfg["ros"]["cube_topic"], 10)
            self.dbg = self.create_publisher(Image, cfg["ros"]["debug_image_topic"], 2)
            self.create_subscription(Image, cfg["ros"]["image_topic"], self.on_image, qos_profile_sensor_data)
            self.prev = None
            self.n = self.miss = 0
            self.get_logger().info(f"cube_detector: {cfg['ros']['image_topic']} -> {cfg['ros']['cube_topic']}")

        def on_image(self, msg: Image) -> None:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            det, _ = detect_cube(img, self.det_cfg, self.calib)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            out = Float64MultiArray()
            self.n += 1
            valid = det is not None
            if valid and self.prev is not None and np.linalg.norm(det.xy - self.prev[:2]) > self.det_cfg["max_jump_m"]:
                valid = False                          # 갑자기 튀는 값 = 오검출로 보고 버림
                self.prev = None                       # 다음 프레임에서 다시 잡음
            if valid:
                a = float(self.det_cfg["smoothing_alpha"])
                xy = det.xy if self.prev is None else a * det.xy + (1 - a) * self.prev[:2]
                yaw = det.yaw                          # yaw 는 ±45° 경계 때문에 평활하지 않음
                self.prev = np.array([xy[0], xy[1], yaw])
                out.data = [float(xy[0]), float(xy[1]), float(yaw), 1.0, stamp]
            else:
                self.miss += 1
                out.data = [0.0, 0.0, 0.0, 0.0, stamp]
            self.pub.publish(out)
            if self.n % 2 == 0:
                self.dbg.publish(self.bridge.cv2_to_imgmsg(draw_detection(img, det), "bgr8"))
            if self.n % 200 == 0:
                self.get_logger().info(f"detection rate {(1 - self.miss / self.n) * 100:.1f}%")

    rclpy.init(args=ros_args)
    node = CubeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
