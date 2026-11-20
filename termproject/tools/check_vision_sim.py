"""[검증] 실물용 색 검출·호모그래피·시차 보정 코드를 시뮬레이션 top_view 이미지로 시험한다.

시뮬레이션 카메라로 찍은 640x480 이미지에서 체커 꼭짓점 픽셀을 카메라 모델로 계산해 호모그래피를 만들고,
real/vision.py 의 detect_cube 결과를 큐브 참값과 비교한다 (위치 오차 mm, yaw 오차 deg).
실행: MUJOCO_GL=egl python -m tools.check_vision_sim --n 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

from pickplace.env import ROOT, PickPlaceEnv
from pickplace.geometry import fold_quarter

sys.path.insert(0, str(ROOT / "real"))
from vision import detect_cube  # noqa: E402

W, H = 640, 480


def project(env: PickPlaceEnv, p_world: np.ndarray) -> np.ndarray:
    cam = env.cam_top
    cp = env.data.cam_xpos[cam]
    R = env.data.cam_xmat[cam].reshape(3, 3)
    pc = R.T @ (p_world - cp)
    f = 0.5 * H / np.tan(0.5 * np.deg2rad(env.model.cam_fovy[cam]))
    return np.array([0.5 * W + f * pc[0] / -pc[2], 0.5 * H - f * pc[1] / -pc[2]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    env = PickPlaceEnv(render_size=(W, H))
    env.reset(seed=0)
    world = np.array([[0.270 + 0.06 * i, 0.06 * j] for i in (-2, 0, 2) for j in (-2, 0, 2)])
    px = np.array([project(env, np.array([x, y, env.surface_z])) for x, y in world])
    Hm, _ = cv2.findHomography(px, world, 0)
    cam = env.data.cam_xpos[env.cam_top]
    calib_plain = {"H": Hm}
    calib_par = {"H": Hm, "camera_height_m": float(cam[2] - env.surface_z), "camera_nadir_xy": cam[:2].tolist()}
    cfg = {"hsv_ranges": [{"low": [40, 80, 60], "high": [85, 255, 255]}], "min_area_px": 30,
           "max_area_px": 5000, "cube_top_height_m": 0.025}
    errs = {"plain": [], "parallax": [], "yaw": []}
    rng = np.random.default_rng(1)
    for k in range(args.n):
        xy = np.array([rng.uniform(0.25, 0.42), rng.uniform(-0.07, 0.07)])
        env.reset(seed=k, options={"cube_xy": xy, "cube_yaw": rng.uniform(-np.pi, np.pi)})
        env.data.qpos[env.jq[1]] = -1.2      # 팔을 시야 밖으로 치움
        mujoco.mj_forward(env.model, env.data)
        img = cv2.cvtColor(env.render("top_view"), cv2.COLOR_RGB2BGR)
        truth = env.cube_pos()[:2]
        for name, cal in (("plain", calib_plain), ("parallax", calib_par)):
            det, _ = detect_cube(img, cfg, cal)
            if det is None:
                continue
            errs[name].append(np.linalg.norm(det.xy - truth) * 1000)
            if name == "parallax":
                errs["yaw"].append(abs(np.degrees(fold_quarter(det.yaw - env.cube_yaw()))))
    for k, v in errs.items():
        unit = "deg" if k == "yaw" else "mm"
        print(f"{k:9s} n={len(v):3d}  mean {np.mean(v):5.2f} {unit}  max {np.max(v):5.2f} {unit}")


if __name__ == "__main__":
    main()
