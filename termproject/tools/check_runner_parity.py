"""[검증] 실물 루프 코어(PolicyCore)가 시뮬레이션 환경과 '같은 관측·같은 관절 목표'를 만드는지 확인.

시뮬레이션 env를 가짜 로봇·가짜 카메라로 쓰고, 같은 정책을 PolicyCore에도 넣어
매 스텝 관측 벡터와 관절 목표의 차이를 잰다. 차이가 0이면 sim과 real 코드 경로가 같다는 근거가 된다.

실행: python -m tools.check_runner_parity [--run A_sym_s7]
"""
from __future__ import annotations

import argparse

import numpy as np
import yaml

from pickplace.controller import PolicyCore
from pickplace.env import ROOT, PickPlaceEnv, load_config
from pickplace.evaluate import load_policy


class LinearPolicy:
    """정책 대용(결정적). 학습된 정책이 없을 때 사용."""

    def __init__(self, dim: int) -> None:
        self.w = np.random.default_rng(0).normal(0, 0.3, size=(5, dim))

    def predict(self, obs, deterministic=True):
        return np.tanh(self.w @ obs), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--exp", nargs="*", default=["config/exp_A_sym.yaml"])
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    if args.run:
        cfg = yaml.safe_load((ROOT / "runs" / args.run / "run_config.yaml").read_text(encoding="utf-8"))
        model = load_policy(ROOT / "runs" / args.run / "policy_best.zip", cfg["training"]["algorithm"])
    else:
        cfg = load_config(*args.exp)
        model = None
    env = PickPlaceEnv(config=cfg)
    model = model or LinearPolicy(env.observation_space.shape[0])
    core = PolicyCore(cfg, env.kin, model)
    obs, _ = env.reset(seed=3)
    core.reset()
    max_obs = max_q = 0.0
    for _ in range(args.steps):
        act, _ = model.predict(obs, deterministic=True)
        q_env, _ = env.action_to_joint_target(act, env.joint_q())
        q_core, _ = core.mapper(act, env.joint_q())
        max_q = max(max_q, float(np.max(np.abs(q_core - q_env))))
        obs, _, term, trunc, _ = env.step(act)
        # 실물 루프와 같은 입력(관절 측정 + 카메라 측정)만으로 core 가 관측을 다시 만든다
        core.grip_closed_cmd = bool(act[4] > 0)
        core.steps = env.steps
        c = env._obs_cache
        meas_xy = None if c["grasped_est"] else c["cube_est"][:2]
        core_obs = core.observe(env.joint_q(), meas_xy, c["cube_yaw_est"])
        max_obs = max(max_obs, float(np.max(np.abs(core_obs - obs))))
        if term or trunc:
            break
    print(f"max |obs_core - obs_env| = {max_obs:.2e}")
    print(f"max |q_core - q_env|     = {max_q:.2e}")
    print("PARITY OK" if max_obs < 1e-5 and max_q < 1e-9 else "PARITY MISMATCH")


if __name__ == "__main__":
    main()
