"""정책 Action → 관절 목표 변환과 관측 생성 — 시뮬레이션·실물 공용 코어.

PickPlaceEnv(시뮬레이션)와 real/policy_runner.py(실물)가 이 클래스를 똑같이 사용한다.
→ "정책 로딩 → 추론 → 로봇 명령" 경로가 시뮬레이션과 실물에서 같은 코드임을 보고서에서 근거로 쓸 수 있다.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .kinematics import GRIPPER_INDEX, IK_JOINTS, ROLL_INDEX, ArmKinematics
from .observation import CubeEstimator, build_observation


def ready_pose(cfg: dict, kin: ArmKinematics) -> np.ndarray:
    """준비 자세(그리퍼 열림, 제어점이 시작 영역 위 약 8 cm). 시뮬레이션 reset·실물 /pickplace/ready 공용."""
    r = cfg["robot"]
    q0 = np.array([0.0, 0.3, 0.0, 1.3, r["ready_wrist_roll_rad"], r["gripper_open_rad"]])
    q, err = kin.solve(q0, r["ready_tcp_xyz"])
    if err > 2e-3:
        raise RuntimeError(f"ready pose IK error {err * 1000:.1f} mm")
    return q


class ActionMapper:
    """Action(5) = [Δx, Δy, Δz, Δyaw, grip] ∈ [-1,1] → 6관절 목표."""

    def __init__(self, cfg: dict, kin: ArmKinematics) -> None:
        c, r = cfg["control"], cfg["robot"]
        self.kin = kin
        self.scale = np.asarray(c["action_scale"], dtype=float)
        self.low = np.asarray(c["tcp_low"], dtype=float)
        self.high = np.asarray(c["tcp_high"], dtype=float)
        self.max_ik_err = float(c["max_accepted_ik_error_m"])
        self.g_open = float(r["gripper_open_rad"])
        self.g_closed = float(r["gripper_closed_rad"])

    def __call__(self, action, q_meas) -> tuple[np.ndarray, float]:
        a = np.clip(np.asarray(action, dtype=float).reshape(5), -1.0, 1.0)
        q_meas = np.asarray(q_meas, dtype=float)
        gp = self.kin.fk(q_meas)["grasp_point"]
        target = np.clip(gp + a[:3] * self.scale[:3], self.low, self.high)
        q_t, err = self.kin.solve(q_meas, target)
        if err > self.max_ik_err:                     # 도달 불가 요청 → 팔은 제자리
            q_t[:IK_JOINTS] = q_meas[:IK_JOINTS]
        lo, hi = self.kin.ranges[ROLL_INDEX]
        q_t[ROLL_INDEX] = np.clip(q_meas[ROLL_INDEX] + a[3] * self.scale[3], lo, hi)
        q_t[GRIPPER_INDEX] = self.g_closed if a[4] > 0 else self.g_open
        return q_t, err


class PolicyCore:
    """실물 루프용: (관절 측정, 카메라 측정) → 관측 → 정책 → 관절 목표. ROS 비의존."""

    def __init__(self, cfg: dict, kin: ArmKinematics, model) -> None:
        self.cfg = cfg
        self.kin = kin
        self.model = model
        self.mapper = ActionMapper(cfg, kin)
        t = cfg["task"]
        self.rest_z = float(t["surface_z_m"]) + float(t["cube_half_size_m"])
        self.goal_xyz = np.array([*t["goal"]["center_xy"], self.rest_z])
        self.dt = 0.002 * int(cfg["control"]["frame_skip"])
        self.max_steps = int(round(float(t["time_limit_s"]) / self.dt))
        self.variant = cfg["observation"]["variant"]
        self.grasp_detect = float(cfg["robot"]["grasp_detect_rad"])
        self.estimator = CubeEstimator(rest_z=self.rest_z)
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.steps = 0
        self.grip_closed_cmd = False

    def observe(self, q_meas, cube_xy: Optional[np.ndarray], cube_yaw: Optional[float]) -> np.ndarray:
        q = np.asarray(q_meas, dtype=float)
        f = self.kin.fk(q)
        grasped = self.grip_closed_cmd and q[GRIPPER_INDEX] > self.grasp_detect
        cube_xyz, yaw = self.estimator.update(cube_xy, cube_yaw, grasped, f["grasp_point"], f["gripper_yaw"], float(q[GRIPPER_INDEX]))
        self.last = {"grasp_point": f["grasp_point"], "gripper_yaw": f["gripper_yaw"],
                     "cube_est": cube_xyz, "cube_yaw_est": yaw, "grasped_est": grasped}
        return build_observation(
            variant=self.variant, joint_q=q, grasp_point=f["grasp_point"], gripper_yaw=f["gripper_yaw"],
            cube_xyz=cube_xyz, cube_yaw=yaw, goal_xyz=self.goal_xyz, grasped=grasped,
            gripper_cmd_closed=self.grip_closed_cmd, cube_visible=cube_xy is not None,
            time_fraction=1.0 - self.steps / self.max_steps)

    def step(self, q_meas, cube_xy, cube_yaw) -> tuple[np.ndarray, np.ndarray, dict]:
        obs = self.observe(q_meas, cube_xy, cube_yaw)
        action, _ = self.model.predict(obs, deterministic=True)
        q_target, ik_err = self.mapper(action, q_meas)
        self.grip_closed_cmd = bool(action[4] > 0)
        self.steps += 1
        return q_target, np.asarray(action), {"obs": obs, "ik_error": ik_err, **self.last}
