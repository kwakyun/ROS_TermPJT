"""ROS 2 노드: 학습한 RL 정책으로 실물 로봇을 20 Hz로 제어한다 (PC에서 실행).

흐름:  /real/joint_states + /cube_obs  →  PolicyCore(관측 생성 → 정책 추론 → IK)  →  /joint_targets
       (Jetson arm_io 가 /joint_targets 를 받아 관절 속도 제한 램프로 서보에 전달)

서비스 (std_srvs/srv/Trigger)
  /pickplace/ready : 준비 자세로 이동(그리퍼 열림). 회차 사이 초기화에 사용
  /pickplace/start : 회차 시작(30 s 타이머, 로그 파일 생성). 시작 후 사람 개입 금지
  /pickplace/stop  : 즉시 정지(현재 자세 유지)

실행:
  source /opt/ros/humble/setup.bash && source ~/termproject/.venv/bin/activate
  export ROS_DOMAIN_ID=<조 번호>
  python3 real/policy_runner.py --config real/config/real.yaml [--run A_sym_s7]
  ros2 service call /robot/set_torque std_srvs/srv/SetBool "{data: true}"
  ros2 service call /pickplace/ready std_srvs/srv/Trigger
  ros2 service call /pickplace/start std_srvs/srv/Trigger
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pickplace.controller import PolicyCore, ready_pose  # noqa: E402
from pickplace.env import DEFAULT_MODEL  # noqa: E402
from pickplace.evaluate import load_policy  # noqa: E402
from pickplace.kinematics import JOINT_NAMES, ArmKinematics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "real/config/real.yaml"))
    ap.add_argument("--run", default=None)
    args, ros_args = ap.parse_known_args()
    rcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rn = rcfg["runner"]
    run = args.run or rn["run"]
    run_dir = ROOT / "runs" / run
    cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    model = load_policy(run_dir / f"policy_{rn['which']}.zip", cfg["training"]["algorithm"])
    c = cfg["control"]
    kin = ArmKinematics(DEFAULT_MODEL, iterations=c["ik_iterations"], damping=c["ik_damping"],
                        orientation_weight=c["ik_orientation_weight"])
    core = PolicyCore(cfg, kin, model)
    q_ready = ready_pose(cfg, kin)
    sim_hz = 1.0 / core.dt
    if abs(sim_hz - rn["rate_hz"]) > 1e-6:
        raise ValueError(f"rate_hz({rn['rate_hz']}) must equal simulation control rate ({sim_hz})")

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from std_srvs.srv import Trigger

    class PolicyRunner(Node):
        def __init__(self) -> None:
            super().__init__("policy_runner")
            ros = rcfg["ros"]
            self.pub = self.create_publisher(JointState, ros["joint_target_topic"], 10)
            self.create_subscription(JointState, ros["joint_state_topic"], self.on_joint, 10)
            self.create_subscription(Float64MultiArray, ros["cube_topic"], self.on_cube, 10)
            self.create_service(Trigger, "/pickplace/ready", self.srv_ready)
            self.create_service(Trigger, "/pickplace/start", self.srv_start)
            self.create_service(Trigger, "/pickplace/stop", self.srv_stop)
            self.q = None
            self.q_time = 0.0
            self.cube = None
            self.cube_time = 0.0
            self.running = False
            self.t0 = 0.0
            self.log = None
            self.trial = 0
            self.create_timer(1.0 / rn["rate_hz"], self.tick)
            self.get_logger().info(f"policy {run}/policy_{rn['which']}.zip loaded, obs={cfg['observation']['variant']}")

        # ---------------- 입력
        def on_joint(self, msg: JointState) -> None:
            pos = dict(zip(msg.name, msg.position))
            if all(n in pos for n in JOINT_NAMES):
                self.q = np.array([pos[n] for n in JOINT_NAMES])
                self.q_time = time.monotonic()

        def on_cube(self, msg: Float64MultiArray) -> None:
            x, y, yaw, valid = msg.data[:4]
            if valid > 0.5:
                self.cube = (np.array([x, y]), float(yaw))
                self.cube_time = time.monotonic()

        # ---------------- 출력
        def send(self, q: np.ndarray) -> None:
            lim = kin.ranges
            q = np.clip(q, lim[:, 0], lim[:, 1])
            if self.q is not None:                    # 안전: 측정값에서 너무 먼 목표는 잘라서 보냄
                j = rn["max_target_jump_rad"]
                q = np.clip(q, self.q - j, self.q + j)
            m = JointState()
            m.header.stamp = self.get_clock().now().to_msg()
            m.name = list(JOINT_NAMES)
            m.position = [float(v) for v in q]
            self.pub.publish(m)

        # ---------------- 서비스
        def srv_ready(self, req, res):
            self.running = False
            self.send(q_ready)
            res.success, res.message = True, "moving to ready pose (gripper open)"
            return res

        def srv_start(self, req, res):
            if self.q is None or time.monotonic() - self.q_time > rn["joint_timeout_s"]:
                res.success, res.message = False, "no fresh /real/joint_states"
                return res
            if np.max(np.abs(self.q[:5] - q_ready[:5])) > 0.08:
                res.success, res.message = False, "robot is not at ready pose: call /pickplace/ready first"
                return res
            core.reset()
            self.trial += 1
            log_dir = ROOT / rn["log_dir"]
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_trial{self.trial}.csv"
            self.log_file = path.open("w", newline="")
            self.log = csv.writer(self.log_file)
            self.log.writerow(["t", *[f"q_{n}" for n in JOINT_NAMES], "cube_visible", "cube_x", "cube_y",
                               "cube_yaw", "est_x", "est_y", "est_z", "grasped_est",
                               "a_dx", "a_dy", "a_dz", "a_dyaw", "a_grip", *[f"tgt_{n}" for n in JOINT_NAMES],
                               "ik_error"])
            self.t0 = time.monotonic()
            self.running = True
            res.success, res.message = True, f"trial {self.trial} started, log {path.name}"
            self.get_logger().info(res.message)
            return res

        def srv_stop(self, req, res):
            self.finish("stopped by service")
            res.success, res.message = True, "stopped"
            return res

        def finish(self, why: str) -> None:
            if self.running:
                self.running = False
                if self.q is not None:
                    self.send(self.q)                 # 현재 자세 유지
                if self.log:
                    self.log_file.close()
                    self.log = None
                self.get_logger().info(f"trial {self.trial} end: {why}")

        # ---------------- 20 Hz 제어 루프
        def tick(self) -> None:
            if not self.running:
                return
            now = time.monotonic()
            t = now - self.t0
            if t >= rn["time_limit_s"]:
                self.finish("time limit 30 s")
                return
            if self.q is None or now - self.q_time > rn["joint_timeout_s"]:
                self.finish("joint state timeout (safety stop)")
                return
            fresh = self.cube is not None and now - self.cube_time <= rn["cube_timeout_s"]
            cxy, cyaw = (self.cube if fresh else (None, None))
            q_target, action, info = core.step(self.q, cxy, cyaw)
            self.send(q_target)
            if self.log:
                e = info["cube_est"]
                self.log.writerow([round(t, 3), *np.round(self.q, 4), int(fresh),
                                   *(np.round(cxy, 4) if fresh else ["", ""]), round(cyaw, 4) if fresh else "",
                                   *np.round(e, 4), int(info["grasped_est"]), *np.round(action, 3),
                                   *np.round(q_target, 4), round(info["ik_error"] * 1000, 2)])

    rclpy.init(args=ros_args)
    node = PolicyRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish("shutdown")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
