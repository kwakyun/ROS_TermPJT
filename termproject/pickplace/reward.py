"""단계형(staged) 보상.

r = max(단계 보상들) + 정렬 항(sym 전용) - 밀기 패널티 - 행동 비용 (+ 종료 보너스/패널티)

- 단계 보상은 '최댓값'으로 합치기 때문에 앞 단계 보상을 반복해서 모으는 꼼수가 통하지 않는다.
- carry(운반)는 '들린 상태'에서만, release/hold는 '목표 안 + 해제' 상태에서만 준다
  → 밀기 전략, 잡은 채로 목표 위에 머무는 전략을 막는다 (TERM_PROJECT 1.3 반영).
"""
from __future__ import annotations

import numpy as np

from .geometry import align_error, fold_quarter


def staged_reward(s: dict, cfg: dict, variant: str) -> tuple[float, dict]:
    """s: 환경이 넘겨주는 한 스텝 상태(시뮬레이션 참값 사용 가능)."""
    d_reach = float(np.linalg.norm(s["grasp_point"] - s["cube_pos"]))
    d_goal_xy = float(np.linalg.norm(s["cube_pos"][:2] - s["goal_xy"]))
    stage = {
        "reach": cfg["reach_scale"] * (1.0 - np.tanh(10.0 * d_reach)),
        "pre_grasp": (0.5 * (cfg["reach_scale"] + cfg["grasp"])
                      if (d_reach < 0.015 and s.get("grip_closed_cmd", False)
                          and not s["grasped"] and not s["ever_lifted"])
                      else 0.0),
        "grasp": cfg["grasp"] if s["grasped"] else 0.0,
        "lift": (cfg["lift"] * (0.7 + 0.3 * min(s["clearance"] / 0.03, 1.0))
                 if s["lifted"] else 0.0),
        "carry": (cfg["carry"] * (0.6 + 0.4 * (1.0 - np.tanh(10.0 * d_goal_xy)))
                  if s["lifted"] else 0.0),
        "place": cfg["place"] if (s["grasped"] and s["on_floor"] and s["inside_goal"]) else 0.0,
        "place_near": (cfg["carry"] * (0.6 + 0.4 * (1.0 - np.tanh(10.0 * d_goal_xy)))
                       if (s["grasped"] and s["on_floor"] and s["ever_lifted"] and not s["inside_goal"])
                       else 0.0),
        "opening": (cfg["place"] + (cfg["release"] - cfg["place"]) * s.get("gripper_open_frac", 0.0)
                    if (s["ever_lifted"] and s["on_floor"] and s["inside_goal"]
                        and not s.get("grip_closed_cmd", True))
                    else 0.0),
        "release": (cfg["release"] + cfg["hold"] * s["hold_fraction"]
                    if (s["released"] and s["on_floor"] and s["inside_goal"] and s["ever_lifted"])
                    else 0.0),
    }
    terms = {"stage": float(max(stage.values()))}
    if variant == "sym":
        w = cfg["align_weight"]
        e_grasp = align_error(s["cube_yaw"], s["gripper_yaw"])
        near = d_reach < 0.04 and not s["grasped"]
        terms["align_grasp"] = w * np.cos(4.0 * e_grasp) if near else 0.0
        carrying_near = s["lifted"] and d_goal_xy < 0.04
        terms["align_place"] = w * np.cos(4.0 * fold_quarter(s["cube_yaw"])) if carrying_near else 0.0
    terms["closed_far"] = (-cfg.get("closed_far_penalty", 0.05)
                           if (s.get("grip_closed_cmd", False) and not s["grasped"]
                               and not s["ever_lifted"] and d_reach > 0.025)
                           else 0.0)
    terms["push"] = -cfg["push_penalty_per_m"] * s["push_displacement"]
    terms["action"] = -cfg["action_cost"] * float(np.sum(np.square(s["action"][:4])))
    terms["success"] = cfg["success_bonus"] if s["success"] else 0.0
    terms["failure"] = -cfg["failure_penalty"] if s["failure"] else 0.0
    terms["_stage_name"] = max(stage, key=stage.get)  # 디버그용(합산 제외)
    total = sum(v for k, v in terms.items() if not k.startswith("_"))
    return float(total), terms
