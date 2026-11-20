"""yaw 대칭 인코딩, 큐브 꼭짓점·목표 영역 판정, 공통 성공 조건 상태기계.

시뮬레이션 환경과 실물 기록 도구가 같은 판정 코드를 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

QUARTER = np.pi / 2.0


# ---------------------------------------------------------------- yaw 유틸
def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def fold_quarter(a: float) -> float:
    """90° 대칭 각도를 [-45°, 45°) 로 접는다. 큐브 yaw θ 와 θ+90° 는 같은 값이 된다."""
    return float((a + np.pi / 4) % QUARTER - np.pi / 4)


def align_error(cube_yaw: float, gripper_yaw: float) -> float:
    """잡기 정렬 오차: 그리퍼 닫힘 축을 큐브 면 법선에 맞추려면 손목을 얼마나 돌려야 하는가."""
    return fold_quarter(cube_yaw - gripper_yaw)


def yaw_features(yaw: float, symmetric: bool) -> np.ndarray:
    """기본: (sin θ, cos θ)  /  개선: (sin 4θ, cos 4θ) — 90° 주기라 θ와 θ+90°가 같은 입력."""
    k = 4.0 if symmetric else 1.0
    return np.array([np.sin(k * yaw), np.cos(k * yaw)], dtype=np.float64)


# ---------------------------------------------------------------- 영역 판정
def bottom_corners(center: np.ndarray, rot: np.ndarray, half: float) -> np.ndarray:
    """큐브 8개 꼭짓점 중 z가 낮은 4개(밑면) XYZ."""
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(center + rot @ (half * np.array([sx, sy, sz], dtype=float)))
    corners = np.asarray(corners)
    return corners[np.argsort(corners[:, 2])[:4]]


def region_margin(points_xy: np.ndarray, center_xy, half_size: float) -> float:
    """정사각형 경계까지의 최소 여유 [m]. 양수=모두 안쪽, 음수=하나라도 밖."""
    rel = np.abs(np.asarray(points_xy)[:, :2] - np.asarray(center_xy)[None, :])
    return float(np.min(half_size - rel))


def yaw_from_rot(rot: np.ndarray) -> float:
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


# ---------------------------------------------------------------- 성공 판정
STAGE_NAMES = ("approach", "grasped", "lifted", "placed", "released", "success")


@dataclass
class StepFacts:
    """한 제어 스텝 끝에서 관측한 사실(시뮬레이션은 접촉으로, 실물은 평가자가 판정)."""
    both_fingers_contact: bool      # 양쪽 손가락이 큐브와 접촉
    cube_on_floor: bool             # 큐브-작업면 접촉
    bottom_clearance_m: float       # 밑면 최저점 높이 - 작업면
    robot_cube_contact: bool        # 로봇 어느 부분이든 큐브와 접촉
    gripper_open: bool              # 그리퍼가 열림 명령 + 열림 각도 이상
    goal_margin_m: float            # 밑면 4꼭짓점의 목표 경계 여유


@dataclass
class SuccessTracker:
    """TERM_PROJECT 1.3 Advanced 성공 조건을 단계 순서대로 판정한다.

    approach → grasped(양 손가락 접촉) → lifted(잡힌 채 밑면이 떨어짐)
    → placed(잡힌 채 다시 작업면에 닿음) → released(완전 해제)
    → success(밑면 전체가 목표 안 + 해제 상태로 hold_time 유지)
    잡지 않은 채 바닥에 떨어지면 dropped 로 기록하고 approach 로 되돌린다 → 밀기·떨어뜨리기는 성공 불가.
    """
    dt: float
    hold_time: float = 2.0
    lift_clearance: float = 0.002
    stage: int = 0
    max_stage: int = 0
    hold_elapsed: float = 0.0
    drops: int = 0
    ever_lifted: bool = False
    events: list = field(default_factory=list)

    def reset(self) -> None:
        self.stage = self.max_stage = 0
        self.hold_elapsed = 0.0
        self.drops = 0
        self.ever_lifted = False
        self.events = []

    def update(self, f: StepFacts, t: float) -> int:
        grasped = f.both_fingers_contact
        lifted = grasped and (not f.cube_on_floor) and f.bottom_clearance_m > self.lift_clearance
        released = (not f.robot_cube_contact) and f.gripper_open
        s = self.stage
        if s in (0, 1):
            if grasped:
                s = 1
            elif s == 1 and not f.robot_cube_contact:
                s = 0                                  # 놓쳐 버림
            if s == 1 and lifted:
                s = 2
                self.ever_lifted = True
        elif s == 2:
            if f.cube_on_floor and grasped:
                s = 3                                  # 잡은 채 내려놓음
            elif f.cube_on_floor and not f.robot_cube_contact:
                s = 0                                  # 운반 중 낙하
                self.drops += 1
                self.events.append((round(t, 2), "dropped"))
        elif s == 3:
            if lifted:
                s = 2                                  # 다시 들어 올림(재배치 허용)
            elif released and f.cube_on_floor:
                s = 4
        elif s == 4:
            if grasped and lifted:
                s = 2
        if s == 4:
            if released and f.cube_on_floor and f.goal_margin_m >= 0.0:
                self.hold_elapsed += self.dt
            else:
                self.hold_elapsed = 0.0
            if self.hold_elapsed + 1e-9 >= self.hold_time:
                s = 5
        else:
            self.hold_elapsed = 0.0
        if s != self.stage:
            self.events.append((round(t, 2), STAGE_NAMES[s]))
        self.stage = s
        self.max_stage = max(self.max_stage, s)
        return s

    @property
    def success(self) -> bool:
        return self.stage == 5

    def end_reason(self, out_of_area: bool, timeout: bool) -> str:
        if self.success:
            return "success"
        if out_of_area:
            return "out_of_area"
        if self.drops and self.stage == 0:
            return "dropped"
        return {
            0: "grasp_fail", 1: "lift_fail", 2: "timeout_carrying",
            3: "timeout_not_released", 4: "outside_goal",
        }.get(self.stage, "timeout") if timeout else "running"
