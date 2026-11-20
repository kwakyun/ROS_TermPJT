"""로봇 기구학(FK/IK) — 시뮬레이션 환경과 실물 policy_runner가 같은 코드를 사용한다.

핵심 정의 (models/robot.xml 의 gripperframe site 기준, 실측으로 확인)
- site +X : 접근 축 (손목 → 손가락 끝). Top-down 자세에서는 world -Z 를 향해야 함
- site +Z : 그리퍼 닫힘 축 (고정 손가락 → 움직이는 손가락 방향)
- gripperframe 원점은 '고정 손가락 끝' 근처이다.
- 제어점(grasp point) = gripperframe + 18 mm(+Z) - 5 mm(+X). 고정 손가락 안쪽 면이 site z=0 이므로
  큐브 반변 12.5 mm + 여유 5.5 mm. 닫으면 움직이는 손가락이 큐브를 고정 손가락 쪽으로 밀어 붙여 잡는다.
  (실측: 그리퍼 0.55 rad 에서 손가락 간격 약 50 mm, 0.10 rad 에서 23 mm)

IK: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex 4관절로
    (1) 제어점 위치 3개 + (2) 접근 축이 아래를 향하는 자세를 감쇠 최소자승(DLS)으로 푼다.
    wrist_roll(그리퍼 yaw)과 gripper는 IK 밖에서 직접 명령한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
IK_JOINTS = 4
ROLL_INDEX = 4
GRIPPER_INDEX = 5
GRASP_OFFSET_LOCAL = np.array([-0.005, 0.0, 0.018])   # gripperframe 기준 (고정 손가락 안쪽 면에서 18 mm)
DOWN = np.array([0.0, 0.0, -1.0])


def wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class ArmKinematics:
    """MuJoCo 모델을 '계산기'로만 쓰는 FK/IK. 물리 스텝은 진행하지 않는다."""

    def __init__(self, model_path: str | Path | None = None,
                 model: mujoco.MjModel | None = None,
                 iterations: int = 40, damping: float = 0.02,
                 orientation_weight: float = 0.08) -> None:
        if model is None:
            model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model = model
        self.data = mujoco.MjData(model)
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.w_ori = float(orientation_weight)
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        if min(jid) < 0:
            raise ValueError("robot joints not found in model")
        self.qadr = model.jnt_qposadr[jid].astype(int)
        self.dadr = model.jnt_dofadr[jid].astype(int)
        self.ranges = model.jnt_range[jid].copy()
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ------------------------------------------------------------------ FK
    def _set(self, q: Sequence[float]) -> None:
        self.data.qpos[self.qadr] = np.asarray(q, dtype=float)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def site_pose(self, q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        self._set(q)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    @staticmethod
    def grasp_point_from_pose(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
        return pos + rot @ GRASP_OFFSET_LOCAL

    @staticmethod
    def gripper_yaw_from_rot(rot: np.ndarray) -> float:
        """닫힘 축(site +Z)의 수평면 방향각. 그리퍼는 180° 대칭이므로 [-π/2, π/2)로 접는다."""
        axis = rot[:, 2]
        yaw = float(np.arctan2(axis[1], axis[0]))
        return float((yaw + np.pi / 2) % np.pi - np.pi / 2)

    def fk(self, q: Sequence[float]) -> dict:
        pos, rot = self.site_pose(q)
        return {
            "tcp": pos,
            "grasp_point": self.grasp_point_from_pose(pos, rot),
            "gripper_yaw": self.gripper_yaw_from_rot(rot),
            "approach_down_cos": float(rot[:, 0] @ DOWN),
        }

    # ------------------------------------------------------------------ IK
    def solve(self, q_now: Sequence[float], target_grasp_point: Sequence[float]
              ) -> tuple[np.ndarray, float]:
        """4관절 IK. 반환: (6관절 목표, 위치 오차[m]). roll·gripper는 q_now 값을 유지."""
        q = np.asarray(q_now, dtype=float).copy()
        target = np.asarray(target_grasp_point, dtype=float)
        err_norm = np.inf
        dofs = self.dadr[:IK_JOINTS]
        for _ in range(self.iterations):
            self._set(q)
            pos = self.data.site_xpos[self.site]
            rot = self.data.site_xmat[self.site].reshape(3, 3)
            point = pos + rot @ GRASP_OFFSET_LOCAL
            e_pos = target - point
            e_ori = np.cross(rot[:, 0], DOWN)          # 접근 축을 아래로 돌리는 회전 벡터
            err_norm = float(np.linalg.norm(e_pos))
            if err_norm < 2e-4 and np.linalg.norm(e_ori) < 2e-3:
                break
            self._jacp[:] = 0.0
            self._jacr[:] = 0.0
            mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, point,
                          self.model.site_bodyid[self.site])
            jac = np.vstack((self._jacp[:, dofs], self.w_ori * self._jacr[:, dofs]))
            err = np.concatenate((e_pos, self.w_ori * e_ori))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + self.damping ** 2 * np.eye(6), err)
            dq = np.clip(dq, -0.15, 0.15)
            q[:IK_JOINTS] = np.clip(q[:IK_JOINTS] + dq,
                                    self.ranges[:IK_JOINTS, 0], self.ranges[:IK_JOINTS, 1])
        self._set(q)
        pos = self.data.site_xpos[self.site]
        rot = self.data.site_xmat[self.site].reshape(3, 3)
        err_norm = float(np.linalg.norm(target - (pos + rot @ GRASP_OFFSET_LOCAL)))
        return q, err_norm
