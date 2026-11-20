"""천장 카메라 → base_link 호모그래피 보정 도구 (OpenCV 창에서 클릭).

작업판 체커 한 칸 = 60 mm, 빨간 인쇄 기준점 = (0.270, 0.000) 은 체커 꼭짓점이다.
따라서 체커 꼭짓점 좌표는 x = 0.270 + 0.060·i, y = 0.060·j 로 정확히 알 수 있다.

사용
 1) 이미지 준비: Jetson 에서 /top_cam 한 장 저장
      python3 real/calibrate_homography.py --grab /tmp/topcam.png      (ROS 환경에서 실행)
 2) 보정:  python3 real/calibrate_homography.py --image /tmp/topcam.png --out real/config/homography.yaml
    창에 표시되는 순서대로 꼭짓점을 클릭한다 (u: 되돌리기, s: 계산·저장, q: 종료).
 3) 확인:  python3 real/calibrate_homography.py --image /tmp/topcam.png --check real/config/homography.yaml
    체커 격자를 다시 그려 겹치는지 보고, 클릭한 점의 재투영 오차(mm)를 출력한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

# 기본 클릭 순서: 빨간 점 주변 체커 꼭짓점 (위·아래·왼쪽·오른쪽은 과제 안내 탑뷰 기준: 위 +X, 왼쪽 +Y)
DEFAULT_POINTS = [
    (0.270, 0.000), (0.390, 0.000), (0.150, 0.000),
    (0.270, 0.120), (0.270, -0.120), (0.390, 0.120), (0.390, -0.120),
    (0.150, 0.120), (0.150, -0.120), (0.450, 0.060), (0.450, -0.060),
]
# 시뮬레이션 top_view 카메라 위치(참고값). 실물은 줄자로 재서 바꾼다.
SIM_CAMERA_NADIR = (0.29084, 0.03815)
SIM_CAMERA_HEIGHT = 0.5091 + 0.0048


def grab_frame(path: str, topic: str = "/top_cam") -> None:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    rclpy.init()
    node = rclpy.create_node("grab_topcam")
    got = {}
    node.create_subscription(Image, topic, lambda m: got.setdefault("img", m), qos_profile_sensor_data)
    while rclpy.ok() and "img" not in got:
        rclpy.spin_once(node, timeout_sec=0.5)
    cv2.imwrite(path, CvBridge().imgmsg_to_cv2(got["img"], "bgr8"))
    print(f"saved {path}")
    node.destroy_node()
    rclpy.shutdown()


def draw_grid(img, H_inv):
    out = img.copy()
    for i in range(-3, 6):
        x = 0.270 + 0.06 * i
        a = cv2.perspectiveTransform(np.array([[[x, -0.24]]]), H_inv)[0, 0]
        b = cv2.perspectiveTransform(np.array([[[x, 0.24]]]), H_inv)[0, 0]
        cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), (0, 255, 255), 1)
    for j in range(-4, 5):
        y = 0.06 * j
        a = cv2.perspectiveTransform(np.array([[[0.09, y]]]), H_inv)[0, 0]
        b = cv2.perspectiveTransform(np.array([[[0.51, y]]]), H_inv)[0, 0]
        cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), (0, 255, 255), 1)
    # 목표 영역 50x50 mm
    sq = np.array([[[0.365, -0.025]], [[0.415, -0.025]], [[0.415, 0.025]], [[0.365, 0.025]]])
    cv2.polylines(out, [cv2.perspectiveTransform(sq, H_inv).astype(int)], True, (0, 0, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grab", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default="real/config/homography.yaml")
    ap.add_argument("--check", default=None)
    ap.add_argument("--camera-height", type=float, default=SIM_CAMERA_HEIGHT,
                    help="작업판에서 카메라 렌즈까지 높이[m] (줄자로 측정)")
    ap.add_argument("--camera-nadir", type=float, nargs=2, default=SIM_CAMERA_NADIR,
                    help="카메라 바로 아래 작업판 점의 base_link XY [m]")
    args = ap.parse_args()
    if args.grab:
        grab_frame(args.grab)
        return
    img = cv2.imread(args.image)
    if args.check:
        d = yaml.safe_load(Path(args.check).read_text(encoding="utf-8"))
        H = np.asarray(d["H"])
        cv2.imshow("check", draw_grid(img, np.linalg.inv(H)))
        print(f"재투영 오차: 평균 {d['rms_error_mm']:.2f} mm, 최대 {d['max_error_mm']:.2f} mm")
        cv2.waitKey(0)
        return

    clicks: list[tuple[float, float]] = []
    win = "click checker corners in order"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(DEFAULT_POINTS):
            clicks.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = img.copy()
        for k, (u, v) in enumerate(clicks):
            cv2.circle(view, (int(u), int(v)), 4, (0, 0, 255), -1)
            cv2.putText(view, str(k), (int(u) + 5, int(v) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        nxt = DEFAULT_POINTS[len(clicks)] if len(clicks) < len(DEFAULT_POINTS) else None
        msg = f"next #{len(clicks)}: (x={nxt[0]:.3f}, y={nxt[1]:+.3f})" if nxt else "all clicked: press s"
        cv2.putText(view, msg + "   [u]ndo [s]ave [q]uit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u") and clicks:
            clicks.pop()
        elif key == ord("q"):
            return
        elif key == ord("s") and len(clicks) >= 4:
            break
    px = np.array(clicks, dtype=np.float64)
    world = np.array(DEFAULT_POINTS[: len(clicks)], dtype=np.float64)
    H, _ = cv2.findHomography(px, world, 0)
    pred = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(pred - world, axis=1) * 1000
    print("점별 재투영 오차 [mm]:", np.round(err, 2))
    data = {
        "H": H.tolist(), "image_size": [int(img.shape[1]), int(img.shape[0])],
        "points_px": px.tolist(), "points_base": world.tolist(),
        "rms_error_mm": float(np.sqrt(np.mean(err ** 2))), "max_error_mm": float(err.max()),
        "camera_height_m": float(args.camera_height), "camera_nadir_xy": list(map(float, args.camera_nadir)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"saved {args.out}  (RMS {data['rms_error_mm']:.2f} mm)")
    cv2.imshow("check", draw_grid(img, np.linalg.inv(H)))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
