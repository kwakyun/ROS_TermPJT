"""Advanced 단일 큐브 Pick-and-place Gymnasium 환경 (7조 텀프로젝트).

- 공통 시작 조건: Seed로 큐브 중심 X[0.270,0.330], Y[-0.060,0.060], yaw[-π,π] 균등 추출,
  빈 그리퍼 준비 자세, 로봇-큐브 비접촉 확인, 물리 시간 30초 상한.
- 공통 성공 조건: geometry.SuccessTracker (집기→들기→내려놓기→해제→밑면 전체 목표 안 2초 유지)
- Action(5): [Δx, Δy, Δz, Δyaw(손목 roll), 그리퍼(>0 닫기)] ∈ [-1,1]
  → 제어점 목표 → kinematics.ArmKinematics IK → 관절 목표 → 속도 제한(실물 arm_io와 같은 램프)
- 관측: observation.build_observation (base / sym) — 실물과 같은 함수
- Domain Randomization: cfg.randomization.enabled
"""
from __future__ import annotations

import copy
from collections import deque
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from .geometry import (StepFacts, SuccessTracker, bottom_corners, fold_quarter,
                       region_margin, wrap_pi, yaw_from_rot)
from .controller import ActionMapper, ready_pose
from .kinematics import GRIPPER_INDEX, JOINT_NAMES, ArmKinematics
from .observation import CubeEstimator, build_observation, make_space
from .reward import staged_reward

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "task.yaml"
DEFAULT_MODEL = ROOT / "models" / "pickplace_scene.xml"


def deep_update(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*paths: str | Path) -> dict:
    """task.yaml 위에 실험 설정(exp_*.yaml)을 순서대로 덮어쓴다."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for p in paths:
        if p:
            with open(p, encoding="utf-8") as f:
                cfg = deep_update(cfg, yaml.safe_load(f) or {})
    return cfg


class PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config: Optional[dict] = None, config_paths: tuple = (),
                 render_mode: Optional[str] = None, render_size=(640, 480)) -> None:
        super().__init__()
        self.cfg = config if config is not None else load_config(*config_paths)
        t, r, c = self.cfg["task"], self.cfg["robot"], self.cfg["control"]
        self.render_mode = render_mode
        self.render_size = render_size
        self.model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
        self.data = mujoco.MjData(self.model)
        self.kin = ArmKinematics(model=self.model, iterations=c["ik_iterations"],
                                 damping=c["ik_damping"],
                                 orientation_weight=c["ik_orientation_weight"])
        m = self.model
        name = lambda kind, n: mujoco.mj_name2id(m, kind, n)  # noqa: E731
        self.jq = self.kin.qadr
        self.act = np.array([name(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES])
        cube_joint = name(mujoco.mjtObj.mjOBJ_JOINT, "object_cube_free")
        self.cube_qadr = int(m.jnt_qposadr[cube_joint])
        self.cube_dadr = int(m.jnt_dofadr[cube_joint])
        self.cube_body = name(mujoco.mjtObj.mjOBJ_BODY, "object_cube")
        self.cube_geom = name(mujoco.mjtObj.mjOBJ_GEOM, "object_cube_geom")
        self.floor_geom = name(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.cam_top = name(mujoco.mjtObj.mjOBJ_CAMERA, "top_view")
        # 손가락 충돌 geom: 고정 손가락(wrist_roll_follower 메시), 움직이는 손가락(moving_jaw 메시)
        mesh_of = lambda g: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])  # noqa: E731
                             if m.geom_dataid[g] >= 0 else "")
        coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.fixed_jaw = {g for g in coll if mesh_of(g) == "wrist_roll_follower_so101_v1"}
        self.moving_jaw = {g for g in coll if mesh_of(g) == "moving_jaw_so101_v1"}
        robot_root = name(mujoco.mjtObj.mjOBJ_BODY, "base")
        self.robot_geoms = {g for g in coll if self._is_descendant(m.geom_bodyid[g], robot_root)}
        if not self.fixed_jaw or not self.moving_jaw:
            raise RuntimeError("finger collision geoms not found")
        sim = self.cfg.get("sim", {})
        m.opt.noslip_iterations = int(sim.get("noslip_iterations", 0))
        if "cube_solref" in sim:
            m.geom_solref[self.cube_geom] = sim["cube_solref"]
        if "gripper_forcerange_nm" in sim:
            fr = float(sim["gripper_forcerange_nm"])
            m.actuator_forcerange[self.act[GRIPPER_INDEX]] = [-fr, fr]
        self._nominal = {
            "cube_friction": m.geom_friction[self.cube_geom].copy(),
            "cube_mass": float(m.body_mass[self.cube_body]),
            "kp": m.actuator_gainprm[self.act, 0].copy(),
            "biasprm": m.actuator_biasprm[self.act, :3].copy(),
        }
        # 상수
        self.dt = m.opt.timestep * c["frame_skip"]
        self.metadata["render_fps"] = int(round(1.0 / self.dt))
        self.max_steps = int(round(t["time_limit_s"] / self.dt))
        self.surface_z = float(t["surface_z_m"])
        self.half = float(t["cube_half_size_m"])
        self.rest_z = self.surface_z + self.half
        self.goal_xy = np.array(t["goal"]["center_xy"], dtype=float)
        self.goal_half = float(t["goal"]["half_size_m"])
        self.goal_xyz = np.array([*self.goal_xy, self.rest_z])
        self.act_scale = np.array(c["action_scale"], dtype=float)
        self.tcp_low = np.array(c["tcp_low"], dtype=float)
        self.tcp_high = np.array(c["tcp_high"], dtype=float)
        self.vmax_nominal = np.array(r["max_joint_vel_rad_s"], dtype=float)
        self.obs_variant = self.cfg["observation"]["variant"]
        self.rew_variant = self.cfg["reward"]["variant"]
        self.observation_space = make_space(self.obs_variant)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.mapper = ActionMapper(self.cfg, self.kin)
        self.tracker = SuccessTracker(dt=self.dt, hold_time=t["hold_time_s"],
                                      lift_clearance=t["lift_clearance_m"])
        self.estimator = CubeEstimator(rest_z=self.rest_z)
        self.curriculum_scale = 1.0
        self._renderer = None
        self._ready_q = None

    @staticmethod
    def _is_descendant_static(m, body, root):
        while body > 0:
            if body == root:
                return True
            body = m.body_parentid[body]
        return body == root

    def _is_descendant(self, body, root):
        return self._is_descendant_static(self.model, body, root)

    # ================================================================ 시작 조건
    def set_curriculum(self, scale: float) -> None:
        self.curriculum_scale = float(np.clip(scale, 0.0, 1.0))

    def _sample_start(self) -> tuple[np.ndarray, float]:
        sp = self.cfg["task"]["spawn"]
        s = self.curriculum_scale
        def rng(lo_hi):
            lo, hi = lo_hi
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * s
            return self.np_random.uniform(mid - half, mid + half)
        return np.array([rng(sp["x"]), rng(sp["y"])]), float(rng(sp["yaw"]))

    def _ready_pose(self) -> np.ndarray:
        if self._ready_q is None:
            self._ready_q = ready_pose(self.cfg, self.kin)
        return self._ready_q.copy()

    def _apply_randomization(self) -> None:
        m, nom, dr = self.model, self._nominal, self.cfg["randomization"]
        m.geom_friction[self.cube_geom] = nom["cube_friction"]
        m.body_mass[self.cube_body] = nom["cube_mass"]
        m.actuator_gainprm[self.act, 0] = nom["kp"]
        m.actuator_biasprm[self.act, :3] = nom["biasprm"]
        self.vmax = self.vmax_nominal.copy()
        self._dr = {"bias_xy": np.zeros(2), "bias_yaw": 0.0, "delay": 0,
                    "noise_xy": 0.0, "noise_yaw": 0.0, "dropout": 0.0,
                    "occlusion": float(self.cfg["observation"].get("occlusion_radius_m", 0.0))}
        if not dr.get("enabled", False):
            return
        u = lambda ab: self.np_random.uniform(*ab)  # noqa: E731
        m.geom_friction[self.cube_geom, 0] = nom["cube_friction"][0] * u(dr["cube_friction_scale"])
        m.body_mass[self.cube_body] = nom["cube_mass"] * u(dr["cube_mass_scale"])
        kp_scale = u(dr["actuator_kp_scale"])
        m.actuator_gainprm[self.act, 0] = nom["kp"] * kp_scale
        m.actuator_biasprm[self.act, 1] = nom["biasprm"][:, 1] * kp_scale
        self.vmax = self.vmax_nominal * u(dr["joint_vel_scale"])
        b = dr["obs_pos_bias_m"]
        lo, hi = dr["obs_delay_steps"]
        self._dr.update({
            "bias_xy": self.np_random.uniform(-b, b, size=2),
            "bias_yaw": u([-dr["obs_yaw_bias_rad"], dr["obs_yaw_bias_rad"]]),
            "delay": int(self.np_random.integers(lo, hi + 1)),
            "noise_xy": dr["obs_pos_noise_m"], "noise_yaw": dr["obs_yaw_noise_rad"],
            "dropout": dr["obs_dropout_prob"], "occlusion": dr["occlusion_radius_m"],
        })

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}
        self._apply_randomization()
        q_ready = self._ready_pose()
        for _ in range(50):
            if "cube_xy" in options:
                xy, yaw = np.asarray(options["cube_xy"], float), float(options.get("cube_yaw", 0.0))
            else:
                xy, yaw = self._sample_start()
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.jq] = q_ready
            self.data.ctrl[self.act] = q_ready
            half = 0.5 * yaw
            self.data.qpos[self.cube_qadr:self.cube_qadr + 7] = (
                xy[0], xy[1], self.rest_z + 0.0003, np.cos(half), 0, 0, np.sin(half))
            mujoco.mj_forward(self.model, self.data)
            if not self._robot_touches_cube():
                break
        else:
            raise RuntimeError("could not create a contact-free start")
        # 큐브를 작업면에 안착시키는 짧은 정착(로봇 정지 상태, 시간은 0으로 되돌림)
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        self.start = {"cube_xy": xy.copy(), "cube_yaw": wrap_pi(yaw)}
        self.cmd = q_ready.copy()            # arm_io 램프 상태(실제로 서보에 보내는 목표)
        self.policy_target = q_ready.copy()  # 정책이 요청한 목표
        self.grip_closed_cmd = False
        self.steps = 0
        self.tracker.reset()
        self.estimator.reset()
        self._meas_queue = deque(maxlen=8)
        self._prev_cube_xy = self.cube_pos()[:2].copy()
        self._last_action = np.zeros(5)
        obs = self._observe()
        return obs, self._info(False, False, "running")

    # ================================================================ 상태 질의
    def joint_q(self) -> np.ndarray:
        return self.data.qpos[self.jq].copy()

    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    def cube_rot(self) -> np.ndarray:
        return self.data.xmat[self.cube_body].reshape(3, 3).copy()

    def cube_yaw(self) -> float:
        return yaw_from_rot(self.cube_rot())

    def _contacts(self) -> dict:
        fixed = moving = floor = robot = False
        for con in self.data.contact[: self.data.ncon]:
            g1, g2 = int(con.geom1), int(con.geom2)
            if self.cube_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.cube_geom else g1
            fixed |= other in self.fixed_jaw
            moving |= other in self.moving_jaw
            floor |= other == self.floor_geom
            robot |= other in self.robot_geoms
        return {"fixed": fixed, "moving": moving, "floor": floor, "robot": robot}

    def _robot_touches_cube(self) -> bool:
        return self._contacts()["robot"]

    def _grasp_flag_from_joint(self) -> bool:
        """실물과 같은 방식의 '잡힘' 추정: 닫힘 명령인데 그리퍼가 다 닫히지 못함."""
        return self.grip_closed_cmd and self.joint_q()[GRIPPER_INDEX] > self.cfg["robot"]["grasp_detect_rad"]

    # ================================================================ 관측
    def _camera_measurement(self, grasp_point: np.ndarray):
        """시뮬레이션 카메라 측정: 참값 + (DR) 편향·잡음·지연·검출 실패·가림."""
        dr = self._dr
        xy = self.cube_pos()[:2] + dr["bias_xy"] + self.np_random.normal(0, dr["noise_xy"] + 1e-12, 2)
        yaw = self.cube_yaw() + dr["bias_yaw"] + self.np_random.normal(0, dr["noise_yaw"] + 1e-12)
        self._meas_queue.append((xy, yaw))
        idx = max(0, len(self._meas_queue) - 1 - dr["delay"])
        xy, yaw = self._meas_queue[idx]
        occluded = dr["occlusion"] > 0 and np.linalg.norm(grasp_point[:2] - self.cube_pos()[:2]) < dr["occlusion"]
        if occluded or self.np_random.random() < dr["dropout"]:
            return None, None
        if self.obs_variant == "sym":
            yaw = fold_quarter(yaw)          # 실물 minAreaRect 처럼 90° 모호성 그대로
        return xy, yaw

    def _observe(self) -> np.ndarray:
        q = self.joint_q()
        f = self.kin.fk(q)
        gp = f["grasp_point"]
        meas_xy, meas_yaw = self._camera_measurement(gp)
        grasped = self._grasp_flag_from_joint()
        cube_xyz, cube_yaw = self.estimator.update(meas_xy, meas_yaw, grasped, gp, f["gripper_yaw"])
        self._obs_cache = {"grasp_point": gp, "gripper_yaw": f["gripper_yaw"],
                           "cube_est": cube_xyz, "cube_yaw_est": cube_yaw, "grasped_est": grasped}
        return build_observation(
            variant=self.obs_variant, joint_q=q, grasp_point=gp, gripper_yaw=f["gripper_yaw"],
            cube_xyz=cube_xyz, cube_yaw=cube_yaw, goal_xyz=self.goal_xyz, grasped=grasped,
            gripper_cmd_closed=self.grip_closed_cmd, cube_visible=meas_xy is not None,
            time_fraction=1.0 - self.steps / self.max_steps)

    # ================================================================ 제어
    def action_to_joint_target(self, action: np.ndarray, q_meas: np.ndarray) -> tuple[np.ndarray, float]:
        """정책 Action → 6관절 목표 (controller.ActionMapper — 실물 policy_runner와 같은 코드)."""
        return self.mapper(action, q_meas)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(5)
        q_meas = self.joint_q()
        self.policy_target, ik_err = self.action_to_joint_target(action, q_meas)
        self.grip_closed_cmd = bool(action[4] > 0)
        step_lim = self.vmax * self.dt
        self.cmd = self.cmd + np.clip(self.policy_target - self.cmd, -step_lim, step_lim)
        self.data.ctrl[self.act] = self.cmd
        mujoco.mj_step(self.model, self.data, nstep=self.cfg["control"]["frame_skip"])
        self.steps += 1

        # ---- 판정용 사실(시뮬레이션 참값)
        con = self._contacts()
        cpos, crot = self.cube_pos(), self.cube_rot()
        corners = bottom_corners(cpos, crot, self.half)
        margin = region_margin(corners, self.goal_xy, self.goal_half)
        clearance = float(corners[:, 2].min() - self.surface_z)
        q = self.joint_q()
        gripper_open = (not self.grip_closed_cmd) and q[GRIPPER_INDEX] > 0.6 * self.cfg["robot"]["gripper_open_rad"]
        facts = StepFacts(both_fingers_contact=con["fixed"] and con["moving"],
                          cube_on_floor=con["floor"], bottom_clearance_m=clearance,
                          robot_cube_contact=con["robot"], gripper_open=gripper_open,
                          goal_margin_m=margin)
        prev_drops = self.tracker.drops
        stage = self.tracker.update(facts, self.data.time)
        dropped_now = self.tracker.drops > prev_drops
        success = self.tracker.success
        tw = self.cfg["task"]["workspace_cube_xy"]
        out = not (tw["x"][0] <= cpos[0] <= tw["x"][1] and tw["y"][0] <= cpos[1] <= tw["y"][1]) \
            or cpos[2] < self.surface_z - 0.01 or cpos[2] > 0.25
        timeout = self.steps >= self.max_steps

        grasped = facts.both_fingers_contact
        lifted = grasped and not con["floor"] and clearance > self.tracker.lift_clearance
        disp = float(np.linalg.norm(cpos[:2] - self._prev_cube_xy))
        push = disp if (con["robot"] and not grasped) else 0.0
        self._prev_cube_xy = cpos[:2].copy()
        reward, terms = staged_reward({
            "grasp_point": self.kin.fk(q)["grasp_point"], "cube_pos": cpos, "goal_xy": self.goal_xy,
            "grasped": grasped, "lifted": lifted, "clearance": clearance, "on_floor": con["floor"],
            "inside_goal": margin >= 0.0, "released": (not con["robot"]) and gripper_open,
            "ever_lifted": self.tracker.ever_lifted,
            "hold_fraction": self.tracker.hold_elapsed / self.tracker.hold_time,
            "grip_closed_cmd": self.grip_closed_cmd,
            "gripper_open_frac": float(np.clip(q[GRIPPER_INDEX] / (0.6 * self.cfg["robot"]["gripper_open_rad"]), 0.0, 1.0)),
            "cube_yaw": self.cube_yaw(), "gripper_yaw": self.kin.fk(q)["gripper_yaw"],
            "push_displacement": push, "action": action, "success": success, "failure": out,
            "dropped_now": dropped_now,
        }, self.cfg["reward"], self.rew_variant)

        terminated = bool(success or out)
        truncated = bool(timeout and not terminated)
        reason = self.tracker.end_reason(out, timeout or terminated)
        obs = self._observe()
        info = self._info(success, out, reason)
        info.update({"reward_terms": terms, "stage": stage, "ik_error": ik_err, "goal_margin_m": margin})
        self._last_action = action
        return obs, reward, terminated, truncated, info

    def _info(self, success: bool, out: bool, reason: str) -> dict[str, Any]:
        return {
            "is_success": bool(success), "out_of_area": bool(out), "end_reason": reason,
            "stage_name": ("approach", "grasped", "lifted", "placed", "released", "success")[self.tracker.stage],
            "max_stage": self.tracker.max_stage, "drops": self.tracker.drops,
            "sim_time": float(self.data.time), "steps": self.steps,
            "start_xy": self.start["cube_xy"].tolist(), "start_yaw": self.start["cube_yaw"],
            "cube_pos": self.cube_pos().tolist(), "cube_yaw": self.cube_yaw(),
        }

    # ================================================================ 렌더링
    def render(self, camera: str | int = "top_view"):
        if self._renderer is None:
            w, h = self.render_size
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        opt = mujoco.MjvOption()
        self._renderer.update_scene(self.data, camera=camera, scene_option=opt)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
