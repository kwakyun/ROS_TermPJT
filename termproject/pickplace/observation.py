"""관측 구성 — 시뮬레이션과 실물(policy_runner)이 '같은 함수'로 관측 벡터를 만든다.

큐브 위치는 카메라처럼 '측정값(xy, yaw) 또는 None'만 받아서 CubeEstimator가 추정한다.
- 잡지 않았을 때: 측정 xy, z = 작업면 위 휴지 높이(카메라는 높이를 모름)
- 잡았을 때    : 그리퍼가 큐브를 가리므로 '잡는 순간의 오프셋 + 현재 제어점'으로 추정
- 측정 실패    : 마지막 값 유지
→ 실물에서 생기는 가림·검출 실패를 시뮬레이션 관측에서도 똑같이 처리한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from gymnasium import spaces

from .geometry import align_error, fold_quarter, wrap_pi, yaw_features

BASE_SIZE = 28
SYM_EXTRA = 2


def observation_size(variant: str) -> int:
    return BASE_SIZE + (SYM_EXTRA if variant == "sym" else 0)


def make_space(variant: str) -> spaces.Box:
    return spaces.Box(-np.inf, np.inf, shape=(observation_size(variant),), dtype=np.float32)


@dataclass
class CubeEstimator:
    rest_z: float
    max_grasp_offset: float = 0.012
    xy: Optional[np.ndarray] = None
    yaw: float = 0.0
    _grasp_offset: Optional[np.ndarray] = None
    _grasp_yaw_offset: float = 0.0
    visible: bool = False

    def reset(self) -> None:
        self.xy = None
        self.yaw = 0.0
        self._grasp_offset = None
        self.visible = False

    def update(self, meas_xy, meas_yaw, grasped: bool, grasp_point: np.ndarray,
               gripper_yaw: float) -> tuple[np.ndarray, float]:
        have = meas_xy is not None
        self.visible = have
        if grasped:
            if self._grasp_offset is None:             # 잡는 순간 오프셋 고정
                ref = np.asarray(meas_xy) if have else (self.xy if self.xy is not None else grasp_point[:2])
                off = ref - grasp_point[:2]
                n = np.linalg.norm(off)
                if n > self.max_grasp_offset:
                    off = off / n * self.max_grasp_offset
                self._grasp_offset = off
                self._grasp_yaw_offset = wrap_pi((meas_yaw if have else self.yaw) - gripper_yaw)
            self.xy = grasp_point[:2] + self._grasp_offset
            self.yaw = wrap_pi(gripper_yaw + self._grasp_yaw_offset)
            z = float(grasp_point[2])
            return np.array([self.xy[0], self.xy[1], z]), self.yaw
        self._grasp_offset = None
        if have:
            self.xy = np.asarray(meas_xy, dtype=float).copy()
            self.yaw = float(meas_yaw)
        if self.xy is None:
            self.xy = grasp_point[:2].copy()
        return np.array([self.xy[0], self.xy[1], self.rest_z]), self.yaw


def build_observation(*, variant: str, joint_q: np.ndarray, grasp_point: np.ndarray,
                      gripper_yaw: float, cube_xyz: np.ndarray, cube_yaw: float,
                      goal_xyz: np.ndarray, grasped: bool, gripper_cmd_closed: bool,
                      cube_visible: bool, time_fraction: float) -> np.ndarray:
    """관측 벡터 (base 28차원 / sym 30차원). 순서를 바꾸면 학습한 정책을 못 쓴다.

    [0:6]   관절각 6 (pan, lift, elbow, wrist_flex, wrist_roll, gripper)
    [6:9]   제어점(grasp point) XYZ
    [9:11]  그리퍼 yaw (sin 2φ, cos 2φ)  ← 2손가락 그리퍼는 180° 대칭
    [11:14] 큐브 XYZ (추정)
    [14:16] 큐브 yaw  base:(sin θ, cos θ) / sym:(sin 4θ, cos 4θ)
    [16:19] 큐브 - 제어점
    [19:22] 목표 - 큐브
    [22:25] 목표 - 제어점
    [25] 잡힘 추정(그리퍼 각도 기반)  [26] 닫힘 명령  [27] 남은 시간 비율
    sym 추가: [28] 잡기 정렬 오차 fold45(θ-φ)/(π/4)  [29] 놓기 정렬 fold45(θ)/(π/4)
    """
    sym = variant == "sym"
    parts = [
        np.asarray(joint_q, dtype=float),
        grasp_point,
        np.array([np.sin(2 * gripper_yaw), np.cos(2 * gripper_yaw)]),
        cube_xyz,
        yaw_features(cube_yaw, sym),
        cube_xyz - grasp_point,
        goal_xyz - cube_xyz,
        goal_xyz - grasp_point,
        np.array([float(grasped), float(gripper_cmd_closed), float(time_fraction)]),
    ]
    if sym:
        parts.append(np.array([
            align_error(cube_yaw, gripper_yaw) / (np.pi / 4),
            fold_quarter(cube_yaw) / (np.pi / 4),
        ]))
    _ = cube_visible  # 가시성은 추정기에 반영됨(관측 차원은 고정)
    return np.concatenate(parts).astype(np.float32)
