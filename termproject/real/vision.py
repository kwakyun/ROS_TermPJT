"""실물 관측: 색 검출 → 픽셀 → base_link 좌표 (ROS 비의존 순수 함수).

- 색 분할: HSV 범위(여러 개 가능, 빨강처럼 Hue가 0/180을 걸치는 색 대응)
- 큐브 윤곽: 가장 큰 덩어리 → minAreaRect(회전 사각형) → 4꼭짓점
- 호모그래피 H: 작업판 평면의 픽셀 → base_link XY (체커 꼭짓점·빨간 기준점으로 보정)
- 시차 보정: 카메라는 큐브 '윗면'(작업판 위 25 mm)을 보므로 평면 호모그래피 결과가
  카메라 바로 아래점(nadir)에서 바깥으로 밀려 보인다 → (1 - h/H_cam) 비율로 당겨 준다.
- yaw: 변환된 꼭짓점으로 변 방향을 구해 [-45°, 45°)로 접음 (큐브 90° 대칭 → 원래 모호성 없음)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import yaml


@dataclass
class CubeDetection:
    xy: np.ndarray          # base_link [m]
    yaw: float              # [-π/4, π/4)
    pixel: np.ndarray       # 중심 픽셀
    area_px: float
    corners_px: np.ndarray  # 4x2


def load_homography(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    d["H"] = np.asarray(d["H"], dtype=float)
    return d


def pixels_to_base(H: np.ndarray, px: np.ndarray) -> np.ndarray:
    pts = np.asarray(px, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def parallax_correct(xy: np.ndarray, nadir_xy, cam_height: float, point_height: float) -> np.ndarray:
    k = 1.0 - point_height / cam_height
    nadir = np.asarray(nadir_xy, dtype=float)
    return nadir + (np.asarray(xy) - nadir) * k


def color_mask(bgr: np.ndarray, hsv_ranges: list) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for r in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(r["low"], np.uint8), np.array(r["high"], np.uint8))
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def detect_cube(bgr: np.ndarray, cfg: dict, calib: dict) -> tuple[Optional[CubeDetection], np.ndarray]:
    mask = color_mask(bgr, cfg["hsv_ranges"])
    roi = cfg.get("roi_px")                       # [x0, y0, x1, y1] 작업판 바깥 오검출 제거
    if roi:
        m2 = np.zeros_like(mask)
        m2[roi[1]:roi[3], roi[0]:roi[2]] = mask[roi[1]:roi[3], roi[0]:roi[2]]
        mask = m2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cfg["min_area_px"] <= cv2.contourArea(c) <= cfg["max_area_px"]]
    if not contours:
        return None, mask
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    corners_px = cv2.boxPoints(rect)
    corners = pixels_to_base(calib["H"], corners_px)
    h_cam = float(calib.get("camera_height_m", 0.0))
    if h_cam > 0:
        corners = parallax_correct(corners, calib["camera_nadir_xy"], h_cam, float(cfg["cube_top_height_m"]))
    center = corners.mean(axis=0)
    edge = corners[1] - corners[0]
    yaw = float(np.arctan2(edge[1], edge[0]))
    yaw = float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    return CubeDetection(center, yaw, np.asarray(rect[0]), float(cv2.contourArea(c)), corners_px), mask


def draw_detection(bgr: np.ndarray, det: Optional[CubeDetection]) -> np.ndarray:
    out = bgr.copy()
    if det is None:
        cv2.putText(out, "no cube", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out
    cv2.polylines(out, [det.corners_px.astype(int)], True, (0, 255, 0), 2)
    cv2.circle(out, tuple(det.pixel.astype(int)), 3, (0, 0, 255), -1)
    cv2.putText(out, f"x={det.xy[0]:.3f} y={det.xy[1]:+.3f} yaw={np.degrees(det.yaw):+.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out
