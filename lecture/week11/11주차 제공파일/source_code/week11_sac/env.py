"""Fixed-layout two-cube contact environment for Week 11."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import yaml

from .observation import IMAGE_HEIGHT, IMAGE_WIDTH, ObservationBuilder
from .reward import compute_transition_reward


MOVING_TIP_SITE_NAME = "moving_fingertip"
MOVING_JAW_BODY_NAME = "moving_jaw_so101_v1"
MOVING_TIP_LOCAL_POSITION = np.asarray(
    [-0.01029, -0.08195, 0.01887], dtype=np.float64
)
ROBOT_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINT_NAMES = ROBOT_JOINT_NAMES[:5]
IK_JOINT_NAMES = ROBOT_JOINT_NAMES[:3]
WRIST_FLEX_INDEX = ROBOT_JOINT_NAMES.index("wrist_flex")
WRIST_ROLL_INDEX = ROBOT_JOINT_NAMES.index("wrist_roll")
GRIPPER_INDEX = ROBOT_JOINT_NAMES.index("gripper")


def project_directory() -> Path:
    return Path(__file__).resolve().parents[1]


def config_path_for(variant: str) -> Path:
    return project_directory() / "config.yaml"


def model_path() -> Path:
    return project_directory() / "models" / "push_scene.xml"


def _object_id(model: mujoco.MjModel, kind: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, kind, name))
    if object_id < 0:
        raise ValueError(f"MuJoCo model does not contain {name!r}")
    return object_id


def _load_model_with_moving_tip(path: Path) -> mujoco.MjModel:
    """The standalone XML already owns the moving-jaw landmark."""
    return mujoco.MjModel.from_xml_path(str(path))


class MovingJawContactEnv(gym.Env):
    """Three-joint XYZ IK; moving-tip perception and two-cube contact success."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        variant: str = "week11",
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        reward_function=None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if variant != "week11":
            raise ValueError("this standalone package implements week11")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.variant = variant
        self._reward_function = reward_function or compute_transition_reward
        self.environment_version = "direct_xyz15_distance_v1"
        self.render_mode = render_mode
        self.config_path = Path(
            config_path or config_path_for(variant)
        ).resolve()
        self.model_path = Path(model_path or project_directory() / "models" / "push_scene.xml").resolve()
        with self.config_path.open("r", encoding="utf-8") as stream:
            full_config = yaml.safe_load(stream) or {}
        self.full_config = full_config
        self.config = full_config["task"]
        self.reward_config = full_config["reward"]

        self.model = _load_model_with_moving_tip(self.model_path)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)

        self._eef_site_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        self._moving_tip_site_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, MOVING_TIP_SITE_NAME
        )
        self._moving_jaw_body_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, MOVING_JAW_BODY_NAME
        )
        if int(self.model.site_bodyid[self._moving_tip_site_id]) != self._moving_jaw_body_id:
            raise RuntimeError("moving_fingertip is not owned by moving jaw body")
        self._base_site_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "baseframe"
        )
        self._camera_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_view"
        )
        self._floor_geom_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self._object_body_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object_cube"
        )
        self._target_body_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_cube"
        )
        self._object_geom_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_cube_geom"
        )
        self._target_geom_id = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_cube_geom"
        )
        object_joint = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_cube_free"
        )
        target_joint = _object_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_cube_free"
        )
        self._object_qpos_address = int(self.model.jnt_qposadr[object_joint])
        self._target_qpos_address = int(self.model.jnt_qposadr[target_joint])
        self._object_dof_address = int(self.model.jnt_dofadr[object_joint])
        self._target_dof_address = int(self.model.jnt_dofadr[target_joint])

        self._joint_ids = np.asarray([
            _object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ROBOT_JOINT_NAMES
        ], dtype=np.int32)
        self._joint_qpos_addresses = self.model.jnt_qposadr[
            self._joint_ids
        ].astype(np.int32)
        self._joint_dof_addresses = self.model.jnt_dofadr[
            self._joint_ids
        ].astype(np.int32)
        self._actuator_ids = np.asarray([
            _object_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ROBOT_JOINT_NAMES
        ], dtype=np.int32)
        self._ik_joint_ids = self._joint_ids[:len(IK_JOINT_NAMES)]
        self._ik_qpos_addresses = self._joint_qpos_addresses[:len(IK_JOINT_NAMES)]
        self._ik_dof_addresses = self._joint_dof_addresses[:len(IK_JOINT_NAMES)]
        self._ik_joint_ranges = self.model.jnt_range[self._ik_joint_ids].copy()
        self._arm_joint_ranges = self.model.jnt_range[
            self._joint_ids[:len(ARM_JOINT_NAMES)]
        ].copy()

        robot = self.config["robot"]
        self._initial_qpos = np.asarray(robot["initial_qpos"], dtype=np.float64)
        if self._initial_qpos.shape != (len(ROBOT_JOINT_NAMES),):
            raise ValueError("robot.initial_qpos must contain six joint values")
        self._fixed_wrist_flex = float(robot["fixed_wrist_flex_rad"])
        self._fixed_wrist_roll = float(robot["fixed_wrist_roll_rad"])
        self._fixed_gripper = float(robot["fixed_gripper_rad"])
        self._initial_qpos = self._initial_qpos.copy()
        self._initial_qpos[WRIST_FLEX_INDEX] = self._fixed_wrist_flex
        self._initial_qpos[WRIST_ROLL_INDEX] = self._fixed_wrist_roll
        self._initial_qpos[GRIPPER_INDEX] = self._fixed_gripper

        self._surface_z = float(self.config["surface_z_m"])
        self._cube_half_size = float(self.config["object"]["half_size_m"])
        self._cube_center_z = self._surface_z + self._cube_half_size + 0.0005
        self._random_yaw = bool(self.config["object"]["initial_yaw_random"])
        self._push_height = float(self.config["object"]["push_height_m"])
        self._target_region = {
            axis: np.asarray(
                self.config["spawn_regions"]["target_cube"][axis],
                dtype=np.float64,
            )
            for axis in ("x", "y")
        }
        probabilities = self.config["scenario"]["probabilities"]
        self._scenario_names = tuple(probabilities)
        object_scenarios = self.config["spawn_regions"].get("object_scenarios")
        self._scenario_regions = None
        self._object_region = None
        if object_scenarios is not None:
            self._scenario_regions = {
                name: {
                    axis: np.asarray(bounds[axis], dtype=np.float64)
                    for axis in ("x", "y")
                }
                for name, bounds in object_scenarios.items()
            }
            if set(probabilities) != set(self._scenario_regions):
                raise ValueError(
                    "scenario probabilities and object regions must match"
                )
        else:
            object_region = self.config["spawn_regions"]["object_cube"]
            self._object_region = {
                axis: np.asarray(object_region[axis], dtype=np.float64)
                for axis in ("x", "y")
            }
        self._scenario_probabilities = np.asarray([
            float(probabilities[name]) for name in self._scenario_names
        ], dtype=np.float64)
        if not np.isclose(np.sum(self._scenario_probabilities), 1.0):
            raise ValueError("scenario probabilities must sum to 1.0")
        self._scenario_probabilities /= np.sum(self._scenario_probabilities)
        self._straight_max_delta_y = float(
            self.config["scenario"].get("straight_max_abs_delta_y_m", 0.010)
        )
        self._minimum_spawn_distance = float(
            self.config["spawn_regions"]["minimum_center_distance_m"]
        )
        self._max_reset_attempts = int(
            self.config["spawn_regions"]["max_reset_attempts"]
        )

        workspace = self.config["workspace"]
        self._eef_low = np.asarray([
            workspace["eef_x"][0], workspace["eef_y"][0], workspace["eef_z"][0]
        ], dtype=np.float64)
        self._eef_high = np.asarray([
            workspace["eef_x"][1], workspace["eef_y"][1], workspace["eef_z"][1]
        ], dtype=np.float64)
        self._cube_low = np.asarray([
            workspace["cube_x"][0], workspace["cube_y"][0]
        ], dtype=np.float64)
        self._cube_high = np.asarray([
            workspace["cube_x"][1], workspace["cube_y"][1]
        ], dtype=np.float64)

        control = self.config["control"]
        self._action_scale = np.asarray(control["action_scale_m"], dtype=np.float64)
        self._frame_skip = int(control["frame_skip"])
        self._ik_iterations = int(control["ik_iterations"])
        self._ik_damping = float(control["ik_damping"])
        self._max_joint_delta = float(control["max_joint_delta_rad"])
        self._max_initial_ik_error = float(control["max_initial_ik_error_m"])
        self._max_accepted_ik_error = float(control["max_accepted_ik_error_m"])

        episode = self.config["episode"]
        self._max_steps = int(episode["max_steps"])
        self._min_cube_z = self._surface_z + float(
            episode["minimum_cube_z_above_surface_m"]
        )
        self._max_cube_z = self._surface_z + float(
            episode["maximum_cube_z_above_surface_m"]
        )
        self._reward_common = self.config["reward_common"]
        self._behind_distance = float(self.config["object"]["behind_distance_m"])

        self._observation_builder = ObservationBuilder()
        self.observation_space = self._observation_builder.space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        period = self.model.opt.timestep * self._frame_skip
        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = max(1, round(1.0 / period))
        self.action_mode = "base_link_xyz_at_gripperframe"

        self._elapsed_steps = 0
        self._last_ik_error = 0.0
        self._ik_rejected = False
        self._contact_latched = False
        self._contact_substep = -1
        self._scenario_name = ""
        self._last_reset_attempts = 0
        self._renderer: Optional[mujoco.Renderer] = None

    @property
    def scenario_names(self) -> tuple[str, ...]:
        return self._scenario_names

    @property
    def uses_shared_object_region(self) -> bool:
        """Whether scenarios are classified after sampling one object region."""
        return self._object_region is not None

    @property
    def moving_tip_site_id(self) -> int:
        return self._moving_tip_site_id

    @property
    def moving_jaw_body_id(self) -> int:
        return self._moving_jaw_body_id

    def _base_rotation(self) -> np.ndarray:
        return np.asarray(self.data.site_xmat[self._base_site_id]).reshape(3, 3)

    def _base_to_world(self, point: np.ndarray) -> np.ndarray:
        return (
            np.asarray(self.data.site_xpos[self._base_site_id])
            + self._base_rotation() @ np.asarray(point, dtype=np.float64)
        )

    def _world_to_base(self, point: np.ndarray) -> np.ndarray:
        return self._base_rotation().T @ (
            np.asarray(point, dtype=np.float64)
            - np.asarray(self.data.site_xpos[self._base_site_id])
        )

    def _sample_region(self, region: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray([
            self.np_random.uniform(*region["x"]),
            self.np_random.uniform(*region["y"]),
        ], dtype=np.float64)

    def _classify_scenario(
        self, object_xy: np.ndarray, target_xy: np.ndarray
    ) -> str:
        delta_y = float(target_xy[1] - object_xy[1])
        if abs(delta_y) <= self._straight_max_delta_y:
            return "straight"
        return "diagonal_left" if delta_y > 0.0 else "diagonal_right"

    def _place_cube(self, qpos_address: int, dof_address: int,
                    xy: np.ndarray, yaw: float) -> None:
        world = self._base_to_world(
            np.asarray([xy[0], xy[1], self._cube_center_z])
        )
        half_yaw = 0.5 * yaw
        self.data.qpos[qpos_address:qpos_address + 7] = (
            world[0], world[1], world[2],
            np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw),
        )
        self.data.qvel[dof_address:dof_address + 6] = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        requested_scenario = options.get("scenario")
        if requested_scenario is None:
            requested_scenario = str(self.np_random.choice(
                self._scenario_names, p=self._scenario_probabilities
            ))
        if requested_scenario not in self._scenario_names:
            raise ValueError(
                f"unknown scenario {requested_scenario!r}; "
                f"choose one of {self._scenario_names}"
            )
        for attempt in range(1, self._max_reset_attempts + 1):
            if self._scenario_regions is not None:
                object_xy = self._sample_region(
                    self._scenario_regions[requested_scenario]
                )
            else:
                object_xy = self._sample_region(self._object_region)
            target_xy = self._sample_region(self._target_region)
            if np.linalg.norm(target_xy - object_xy) < self._minimum_spawn_distance:
                continue
            if (
                self._object_region is not None
                and self._classify_scenario(object_xy, target_xy)
                != requested_scenario
            ):
                continue
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self._joint_qpos_addresses] = self._initial_qpos
            self.data.ctrl[self._actuator_ids] = self._initial_qpos
            mujoco.mj_forward(self.model, self.data)
            object_yaw = float(self.np_random.uniform(-np.pi, np.pi)) if self._random_yaw else 0.0
            target_yaw = float(self.np_random.uniform(-np.pi, np.pi)) if self._random_yaw else 0.0
            self._place_cube(
                self._object_qpos_address, self._object_dof_address,
                object_xy, object_yaw,
            )
            self._place_cube(
                self._target_qpos_address, self._target_dof_address,
                target_xy, target_yaw,
            )
            mujoco.mj_forward(self.model, self.data)
            if self._cubes_touching() or self._cube_has_invalid_reset_contact():
                continue
            behind = self._behind_point()
            _, initial_error = self._solve_ik(
                np.asarray([behind[0], behind[1], self._push_height])
            )
            if initial_error > self._max_initial_ik_error:
                continue
            self._last_reset_attempts = attempt
            self._scenario_name = requested_scenario
            break
        else:
            raise RuntimeError(
                f"Could not produce a valid reset in {self._max_reset_attempts} attempts"
            )

        self.data.time = 0.0
        self._elapsed_steps = 0
        self._last_ik_error = 0.0
        self._ik_rejected = False
        self._contact_latched = False
        self._contact_substep = -1
        observation = self._build_observation()
        return observation, self._get_info(False, False, "", {}, 0.0)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,):
            raise ValueError(f"action must have shape (3,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action contains NaN or infinity")
        action = np.clip(action, -1.0, 1.0)

        current_eef = self.eef_position()
        target_xyz = np.clip(
            current_eef + action * self._action_scale,
            self._eef_low,
            self._eef_high,
        )
        requested_delta = target_xyz - current_eef
        current_joints = self.data.qpos[self._ik_qpos_addresses].copy()
        accepted = current_joints.copy()
        self._ik_rejected = True
        self._last_ik_error = float("inf")
        for scale in (1.0, 0.5, 0.25, 0.125):
            joint_target, error = self._solve_ik(
                current_eef + scale * requested_delta
            )
            self._last_ik_error = min(self._last_ik_error, error)
            if error > self._max_accepted_ik_error:
                continue
            accepted = np.clip(
                joint_target,
                current_joints - self._max_joint_delta,
                current_joints + self._max_joint_delta,
            )
            self._last_ik_error = error
            self._ik_rejected = False
            break
        self.data.ctrl[self._actuator_ids[:len(IK_JOINT_NAMES)]] = accepted
        self.data.ctrl[self._actuator_ids[WRIST_FLEX_INDEX]] = self._fixed_wrist_flex
        self.data.ctrl[self._actuator_ids[WRIST_ROLL_INDEX]] = self._fixed_wrist_roll
        self.data.ctrl[self._actuator_ids[GRIPPER_INDEX]] = self._fixed_gripper
        self._contact_latched = False
        self._contact_substep = -1
        for substep in range(self._frame_skip):
            mujoco.mj_step(self.model, self.data)
            # Refresh positions and contacts at the end of this physics step.
            mujoco.mj_forward(self.model, self.data)
            if self._cubes_touching():
                self._contact_latched = True
                self._contact_substep = substep + 1
                break

        self._elapsed_steps += 1
        success = self._contact_latched
        failure, failure_reason = self._failure_state()
        transition = {
            "tip_object_distance": float(np.linalg.norm(self.moving_tip_position() - self.cube_position())),
            "object_target_distance": float(np.linalg.norm(self.cube_position() - self.cube_position(True))),
            "success": bool(success),
            "failure": bool(failure),
        }
        reward, reward_terms = self._reward_function(
            transition, self._reward_common, self.reward_config
        )
        terminated = bool(success or failure)
        truncated = bool(self._elapsed_steps >= self._max_steps and not terminated)
        observation = self._build_observation()
        info = self._get_info(
            success, failure, failure_reason, reward_terms, reward
        )
        return observation, reward, terminated, truncated, info

    def _solve_ik(self, target_base_xyz: np.ndarray) -> tuple[np.ndarray, float]:
        self._ik_data.qpos[:] = self.data.qpos
        self._ik_data.qvel[:] = 0.0
        target_world = self._base_to_world(target_base_xyz)
        jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        joint_target = self._ik_data.qpos[self._ik_qpos_addresses].copy()
        error_norm = float("inf")
        for _ in range(self._ik_iterations):
            mujoco.mj_forward(self.model, self._ik_data)
            error = target_world - self._ik_data.site_xpos[self._eef_site_id]
            error_norm = float(np.linalg.norm(error))
            if error_norm < 1.0e-5:
                break
            jacobian[:] = 0.0
            mujoco.mj_jacSite(
                self.model, self._ik_data, jacobian, None, self._eef_site_id
            )
            reduced = jacobian[:, self._ik_dof_addresses]
            damping = self._ik_damping ** 2 * np.eye(3)
            delta = reduced.T @ np.linalg.solve(
                reduced @ reduced.T + damping, error
            )
            delta = np.clip(delta, -0.12, 0.12)
            joint_target = np.clip(
                joint_target + delta,
                self._ik_joint_ranges[:, 0],
                self._ik_joint_ranges[:, 1],
            )
            self._ik_data.qpos[self._ik_qpos_addresses] = joint_target
        return joint_target.copy(), error_norm

    def _cube_has_invalid_reset_contact(self) -> bool:
        cube_geoms = {self._object_geom_id, self._target_geom_id}
        allowed = {self._floor_geom_id}
        for contact in self.data.contact:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in cube_geoms and geom2 not in allowed | cube_geoms:
                return True
            if geom2 in cube_geoms and geom1 not in allowed | cube_geoms:
                return True
        return False

    def _cubes_touching(self) -> bool:
        pair = {self._object_geom_id, self._target_geom_id}
        return any(
            {int(contact.geom1), int(contact.geom2)} == pair
            for contact in self.data.contact
        )

    def _target_has_robot_contact(self) -> bool:
        allowed = {self._floor_geom_id, self._object_geom_id}
        for contact in self.data.contact:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == self._target_geom_id and geom2 not in allowed:
                return True
            if geom2 == self._target_geom_id and geom1 not in allowed:
                return True
        return False

    def cube_position(self, target: bool = False) -> np.ndarray:
        body_id = self._target_body_id if target else self._object_body_id
        return self._world_to_base(self.data.xpos[body_id])

    def eef_position(self) -> np.ndarray:
        """IK control point (`gripperframe`) in base_link coordinates."""
        return self._world_to_base(self.data.site_xpos[self._eef_site_id])

    def moving_tip_position(self) -> np.ndarray:
        """Moving-jaw distal contact landmark in base_link coordinates."""
        return self._world_to_base(
            self.data.site_xpos[self._moving_tip_site_id]
        )

    def _push_direction(self) -> np.ndarray:
        vector = self.cube_position(True)[:2] - self.cube_position(False)[:2]
        return vector / max(float(np.linalg.norm(vector)), 1.0e-9)

    def _behind_point(self) -> np.ndarray:
        return (
            self.cube_position(False)[:2]
            - self._behind_distance * self._push_direction()
        )

    def _failure_state(self) -> tuple[bool, str]:
        for label, position in (
            ("object", self.cube_position(False)),
            ("target", self.cube_position(True)),
        ):
            if np.any(position[:2] < self._cube_low) or np.any(
                position[:2] > self._cube_high
            ):
                return True, f"{label}_outside_workspace"
            if position[2] < self._min_cube_z or position[2] > self._max_cube_z:
                return True, f"{label}_invalid_height"
        return False, ""

    def _project_pixel(self, point_world: np.ndarray) -> np.ndarray:
        camera_position = np.asarray(self.data.cam_xpos[self._camera_id])
        camera_rotation = np.asarray(
            self.data.cam_xmat[self._camera_id]
        ).reshape(3, 3)
        camera_point = camera_rotation.T @ (
            np.asarray(point_world) - camera_position
        )
        depth = -float(camera_point[2])
        if depth <= 1.0e-8:
            return np.asarray([np.nan, np.nan])
        focal = 0.5 * IMAGE_HEIGHT / np.tan(
            0.5 * np.deg2rad(float(self.model.cam_fovy[self._camera_id]))
        )
        return np.asarray([
            0.5 * IMAGE_WIDTH + focal * camera_point[0] / depth,
            0.5 * IMAGE_HEIGHT - focal * camera_point[1] / depth,
        ])

    def eef_pixel(self) -> np.ndarray:
        return self._project_pixel(self.data.site_xpos[self._eef_site_id])

    def moving_tip_pixel(self) -> np.ndarray:
        return self._project_pixel(
            self.data.site_xpos[self._moving_tip_site_id]
        )

    def cube_pixel(self, target: bool = False) -> np.ndarray:
        body_id = self._target_body_id if target else self._object_body_id
        return self._project_pixel(self.data.xpos[body_id])

    def cube_bbox_pixels(self, target: bool = False) -> tuple[np.ndarray, np.ndarray]:
        geom_id = self._target_geom_id if target else self._object_geom_id
        center = np.asarray(self.data.geom_xpos[geom_id])
        rotation = np.asarray(self.data.geom_xmat[geom_id]).reshape(3, 3)
        half = np.asarray(self.model.geom_size[geom_id, :3])
        pixels = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corner = center + rotation @ (half * (sx, sy, sz))
                    pixels.append(self._project_pixel(corner))
        points = np.asarray(pixels)
        return np.nanmin(points, axis=0), np.nanmax(points, axis=0)

    def spawn_region_pixels(
        self, *, target: bool = False, scenario: Optional[str] = None
    ) -> np.ndarray:
        if target:
            region = self._target_region
        else:
            if self._object_region is not None:
                region = self._object_region
            else:
                scenario = scenario or self._scenario_name
                if scenario not in self._scenario_regions:
                    raise ValueError(f"unknown scenario {scenario!r}")
                region = self._scenario_regions[scenario]
        points = []
        for x, y in (
            (region["x"][0], region["y"][0]),
            (region["x"][1], region["y"][0]),
            (region["x"][1], region["y"][1]),
            (region["x"][0], region["y"][1]),
        ):
            points.append(self._project_pixel(self._base_to_world(
                np.asarray([x, y, self._surface_z + 0.001])
            )))
        return np.asarray(points)

    def _build_observation(self) -> np.ndarray:
        return self._observation_builder.build(
            joint_qpos=self.data.qpos[self._joint_qpos_addresses],
            moving_tip_position=self.moving_tip_position(),
            object_position=self.cube_position(),
            target_position=self.cube_position(True),
        )

    def _get_info(
        self,
        success: bool,
        failure: bool,
        failure_reason: str,
        reward_terms: dict[str, float],
        reward: float,
    ) -> dict[str, Any]:
        return {
            "environment_version": self.environment_version,
            "is_success": bool(success),
            "is_failure": bool(failure),
            "failure_reason": failure_reason,
            "elapsed_steps": int(self._elapsed_steps),
            "object_position": self.cube_position(False).copy(),
            "target_position": self.cube_position(True).copy(),
            "moving_tip_position": self.moving_tip_position().copy(),
            "eef_position": self.eef_position().copy(),
            "distance_between_cubes": float(
                np.linalg.norm(
                    self.cube_position(True)
                    - self.cube_position(False)
                )
            ),
            "cubes_touching": bool(self._cubes_touching()),
            "contact_latched": bool(self._contact_latched),
            "contact_substep": int(self._contact_substep),
            "target_robot_contact": self._target_has_robot_contact(),
            "scenario": self._scenario_name,
            "ik_error": float(self._last_ik_error),
            "ik_rejected": bool(self._ik_rejected),
            "reset_attempts": int(self._last_reset_attempts),
            "reward_terms": dict(reward_terms),
            "computed_reward": float(reward),
            "coordinate_frame": "base_link",
            "policy_point": MOVING_TIP_SITE_NAME,
            "ik_control_point": "gripperframe",
        }

    def render_camera(self, camera_name: str = "top_view") -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
            )
        option = mujoco.MjvOption()
        option.geomgroup[4] = 1
        self._renderer.update_scene(self.data, camera=camera_name, scene_option=option)
        return self._renderer.render().copy()

    def render(self) -> Optional[np.ndarray]:
        return self.render_camera("top_view")

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


PushEnv = MovingJawContactEnv
