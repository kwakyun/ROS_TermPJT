"""Show Week 11 Top/Front cameras and a free MuJoCo view."""

from __future__ import annotations

import argparse
import threading
import time

import cv2
import mujoco.viewer
import numpy as np
from stable_baselines3 import SAC

from .env import PushEnv
from .evaluate import resolve_policy_path
from .observation import OBSERVATION_SIZE
from .visualization import front_view_bgr, top_view_bgr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--model", choices=("best", "final"), default="final")
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--delay", type=float, default=0.04)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(1)
    environment = PushEnv(render_mode="rgb_array")
    model = None
    if args.run is not None or args.policy is not None:
        policy_path = resolve_policy_path(args.run or "latest", args.policy, args.model)
        model = SAC.load(str(policy_path), env=environment, device="auto")
        if getattr(model, "environment_version", None) != environment.environment_version:
            raise ValueError("Policy is from an incompatible environment; retrain for direct_xyz15_distance_v1")
        print(f"policy={policy_path}")
    else:
        print("Policy 없이 고정 초기 환경을 확인합니다.")
    print(f"observation={OBSERVATION_SIZE}D, action=XYZ 3D")
    print("R=리셋, Space=일시정지, Q/Esc=종료")
    print("MuJoCo 자유 시점과 Top/Front camera 창을 함께 표시합니다.")

    observation, info = environment.reset(seed=args.seed)
    paused = False
    reset_requested = threading.Event()
    quit_requested = threading.Event()

    def viewer_key(key: int) -> None:
        if key in (ord("r"), ord("R")):
            reset_requested.set()
        elif key in (ord("q"), ord("Q"), 256):
            quit_requested.set()

    def reset_episode() -> None:
        nonlocal observation, info
        observation, info = environment.reset(seed=None)
        cube = info["object_position"]
        print(f"reset cube=({cube[0]:.3f}, {cube[1]:.3f}) initial yaw=0")

    try:
        with mujoco.viewer.launch_passive(
            environment.model, environment.data, key_callback=viewer_key
        ) as viewer:
            viewer.opt.geomgroup[4] = 1
            viewer.cam.lookat[:] = (0.28, 0.0, 0.22)
            viewer.cam.distance = 1.15
            viewer.cam.azimuth = -40.0
            viewer.cam.elevation = -24.0
            while viewer.is_running() and not quit_requested.is_set():
                if reset_requested.is_set():
                    reset_requested.clear()
                    reset_episode()
                action = np.zeros(3, dtype=np.float32)
                reward = 0.0
                done = False
                if model is not None and not paused:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = environment.step(action)
                    done = bool(terminated or truncated)
                frame = top_view_bgr(
                    environment,
                    (
                        "Week11 fixed diagonal | distance reward",
                        f"action XYZ=({action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f})",
                        f"reward={float(reward):+.3f} success={info.get('is_success', False)}",
                    ),
                )
                front_frame = front_view_bgr(environment)
                cv2.imshow("Week11 - Top camera keypoints", frame)
                cv2.imshow("Week11 - Front camera", front_frame)
                viewer.sync()
                key = cv2.waitKey(max(1, int(args.delay * 1000))) & 0xFF
                if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
                       for name in ("Week11 - Top camera keypoints", "Week11 - Front camera")):
                    break
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key in (ord("r"), ord("R")) or done:
                    reset_episode()
                time.sleep(0.001)
    finally:
        environment.close()
        cv2.destroyAllWindows()
        time.sleep(1.0)


if __name__ == "__main__":
    main()
