"""[검증 전용] 손으로 짠 상태기계로 환경을 끝까지 수행해 본다.

목적: (1) IK·그리퍼 각도·Action 스케일로 과제가 '물리적으로 가능한지',
      (2) 성공 판정(SuccessTracker)이 단계별로 맞게 바뀌는지 확인.
※ 평가·제출용 정책이 아니다. 제출 정책은 반드시 RL(train.py)로 학습한 것을 쓴다.

실행: python -m tools.scripted_check --seeds 0 1 2 3 4 5 6 7 8 9 [--video out.mp4]
"""
from __future__ import annotations

import argparse

import numpy as np

from pickplace.env import PickPlaceEnv, load_config
from pickplace.geometry import align_error, fold_quarter


def scripted_action(env: PickPlaceEnv, phase: dict) -> np.ndarray:
    q = env.joint_q()
    f = env.kin.fk(q)
    gp, gyaw = f["grasp_point"], f["gripper_yaw"]
    cube = env.cube_pos()
    a = np.zeros(5)
    p = phase["name"]
    def move_to(target, gain=1.0):
        d = (target - gp) / env.act_scale[:3]
        return np.clip(gain * d, -1, 1)
    e = align_error(env.cube_yaw(), gyaw)
    if p == "above":
        a[:3] = move_to(cube + [0, 0, 0.05]); a[3] = np.clip(e / env.act_scale[3], -1, 1); a[4] = -1
        if np.linalg.norm(gp - (cube + [0, 0, 0.05])) < 0.006 and abs(e) < 0.04:
            phase["name"] = "descend"
    elif p == "descend":
        a[:3] = move_to(cube); a[3] = np.clip(e / env.act_scale[3], -1, 1); a[4] = -1
        if np.linalg.norm(gp - cube) < 0.004:
            phase["name"] = "close"; phase["t"] = 0
    elif p == "close":
        a[4] = 1; phase["t"] += 1
        if phase["t"] > 30:
            phase["name"] = "lift"
    elif p == "lift":
        a[:3] = move_to(np.array([gp[0], gp[1], 0.06])); a[4] = 1
        if gp[2] > 0.055:
            phase["name"] = "carry"
    elif p == "carry":
        tgt = np.array([*env.goal_xy, 0.06]); a[:3] = move_to(tgt); a[4] = 1
        a[3] = np.clip(-fold_quarter(env.cube_yaw()) / env.act_scale[3], -1, 1)   # 목표 축에 맞춰 놓기
        if np.linalg.norm(gp - tgt) < 0.006:
            phase["name"] = "lower"
    elif p == "lower":
        tgt = np.array([*env.goal_xy, env.rest_z + 0.002]); a[:3] = move_to(tgt); a[4] = 1
        if np.linalg.norm(gp - tgt) < 0.004 or env.tracker.stage == 3:   # 작업면에 닿으면 놓기
            phase["name"] = "open"; phase["t"] = 0
    elif p == "open":
        a[4] = -1; phase["t"] += 1
        if phase["t"] > 35:
            phase["name"] = "retreat"
    elif p == "retreat":
        a[:3] = move_to(np.array([gp[0] - 0.01, gp[1], 0.08])); a[4] = -1
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--config", nargs="*", default=[])
    args = ap.parse_args()
    env = PickPlaceEnv(config=load_config(*args.config))
    n_ok = 0
    for s in args.seeds:
        obs, info = env.reset(seed=s)
        phase = {"name": "above", "t": 0}
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(scripted_action(env, phase))
            done = term or trunc
        n_ok += info["is_success"]
        print(f"seed {s}: start=({info['start_xy'][0]:.3f},{info['start_xy'][1]:+.3f}) "
              f"yaw={np.degrees(info['start_yaw']):+6.1f}° -> {info['end_reason']:<20} "
              f"t={info['sim_time']:.1f}s margin={info['goal_margin_m']*1000:+.1f}mm "
              f"events={env.tracker.events}")
    print(f"scripted success {n_ok}/{len(args.seeds)}")


if __name__ == "__main__":
    main()
