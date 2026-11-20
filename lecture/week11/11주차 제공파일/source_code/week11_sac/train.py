"""Train and save the Week 11 SAC policy."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict

import yaml

from .env import PushEnv, project_directory
from .evaluate import evaluate_model


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _format_seconds(seconds: float) -> str:
    minutes, seconds_int = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return (
        f"{hours}h {minutes:02d}m {seconds_int:02d}s"
        if hours else f"{minutes}m {seconds_int:02d}s"
    )


def main() -> None:
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.run_name and Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be a directory name")
    torch.set_num_threads(args.threads)

    root = project_directory()
    config = _load_yaml(root / "config.yaml")
    training = config["training"]
    total_steps = int(args.steps or training["total_timesteps"])
    seed = int(training["seed"] if args.seed is None else args.seed)
    device = str(args.device or training.get("device", "auto"))
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = root / "outputs"
    run_directory = output_root / run_name
    if run_directory.exists():
        raise FileExistsError(f"Run already exists: {run_directory}")
    run_directory.mkdir(parents=True)
    checkpoints = run_directory / "checkpoints"
    tensorboard = run_directory / "tensorboard"
    checkpoints.mkdir()
    tensorboard.mkdir()
    (run_directory / "run_config.yaml").write_text(
        yaml.safe_dump(
            {
                "task": config["task"],
                "reward": config["reward"],
                "training": training,
                "effective": {
                    "total_timesteps": total_steps,
                    "seed": seed,
                    "device": device,
                    "threads": args.threads,
                    "action_mode": "base_link_xyz_at_gripperframe",
                    "environment_version": "direct_xyz15_distance_v1",
                },
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    raw_environment = PushEnv()
    environment = Monitor(
        raw_environment,
        filename=str(run_directory / "training.monitor.csv"),
        info_keywords=("is_success", "scenario", "failure_reason"),
    )
    evaluation_environment = PushEnv()

    class ProgressCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.started = time.monotonic()
            self.next_step = int(training["progress_interval"])

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_step and self.num_timesteps < total_steps:
                return True
            elapsed = time.monotonic() - self.started
            rate = self.num_timesteps / max(elapsed, 1.0e-9)
            eta = (total_steps - self.num_timesteps) / max(rate, 1.0e-9)
            print(
                f"Progress {self.num_timesteps}/{total_steps} "
                f"({100.0*self.num_timesteps/total_steps:.1f}%) | "
                f"elapsed {_format_seconds(elapsed)} | ETA {_format_seconds(eta)}",
                flush=True,
            )
            self.next_step += int(training["progress_interval"])
            return True

    class ValidationCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.success_threshold = float(training["early_stop_success_rate"])
            self.minimum_rollout_episodes = int(
                training.get("rollout_min_episodes", 20)
            )
            self.rollout_gate_armed = True
            self.best_success = -1.0
            self.best_distance = float("inf")
            self.stopped_early = False
            self.stop_step = None
            self.excluded_validation_seconds = 0.0
            self.csv_path = run_directory / "evaluations" / "history.csv"
            self.csv_path.parent.mkdir()
            with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(
                    ["timesteps", "success_rate", "mean_final_distance_m"]
                )

        def _rollout_success_rate(self) -> float | None:
            """Return the same recent success metric shown by SB3 rollout logs."""
            successes = getattr(self.model, "ep_success_buffer", None)
            if not successes or len(successes) < self.minimum_rollout_episodes:
                return None
            return sum(float(value) for value in successes) / len(successes)

        def _on_step(self) -> bool:
            rollout_success = self._rollout_success_rate()
            if rollout_success is None:
                return True
            if rollout_success < self.success_threshold:
                self.rollout_gate_armed = True
                return True
            if not self.rollout_gate_armed:
                return True
            self.rollout_gate_armed = False
            validation_started = time.monotonic()
            summary, _ = evaluate_model(
                model=self.model,
                environment=evaluation_environment,
                episodes_per_scenario=int(
                    training["evaluation_episodes_per_scenario"]
                ),
                seed=int(training["validation_seed"]),
            )
            overall = summary["overall"]
            success = float(overall["success_rate"])
            distance = float(overall["mean_final_distance_m"])
            with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(
                    [self.num_timesteps, success, distance]
                )
            if success > self.best_success or (
                success == self.best_success and distance < self.best_distance
            ):
                self.best_success = success
                self.best_distance = distance
                self.model.save(str(run_directory / "policy_best"))
            should_stop = success >= self.success_threshold
            if should_stop:
                self.stopped_early = True
                self.stop_step = int(self.num_timesteps)
                self.model.save(str(run_directory / "policy_converged"))
            self.excluded_validation_seconds += (
                time.monotonic() - validation_started
            )
            return not should_stop

    validation_callback = ValidationCallback()
    # TODO 2 — SAC()에 MlpPolicy와 environment를 전달하여 model을 생성한다.
    # training의 학습률·Buffer·Batch·gamma·tau·엔트로피·갱신 설정을 연결한다.
    # seed·device, policy_kwargs의 net_arch, tensorboard_log, verbose=1도 지정한다.
    model = SAC(
        "MlpPolicy",
        environment,
        learning_rate=float(training["learning_rate"]),
        buffer_size=int(training["buffer_size"]),
        learning_starts=int(training["learning_starts"]),
        batch_size=int(training["batch_size"]),
        gamma=float(training["gamma"]),
        tau=float(training["tau"]),
        ent_coef=training["ent_coef"],
        train_freq=int(training["train_freq"]),
        gradient_steps=int(training["gradient_steps"]),
        seed=seed,
        device=device,
        policy_kwargs={"net_arch": list(training["network_architecture"])},
        tensorboard_log=str(tensorboard),
        verbose=1,
    )
    model.environment_version = raw_environment.environment_version
    callbacks = CallbackList(
        [
            ProgressCallback(),
            validation_callback,
            CheckpointCallback(
                save_freq=int(training["checkpoint_frequency"]),
                save_path=str(checkpoints),
                name_prefix="sac_push",
            ),
        ]
    )
    started = time.monotonic()
    try:
        print(f"Run: {run_name}")
        print(f"Device: {model.device}")
        # TODO 3 — model.learn()에 total_steps를 total_timesteps로 전달하여 학습한다.
        # callbacks와 training의 log_interval을 연결하고 tb_log_name을 sac_push로 지정한다.
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            log_interval=int(training["log_interval"]),
            tb_log_name="sac_push",
        )
        # TODO 4 — model.save()로 run_directory 아래 policy_final 이름의 최종 정책을 저장한다.
        # Path를 str()로 바꿔 전달하면 policy_final.zip이 생성된다.
        model.save(str(run_directory / "policy_final"))
        (output_root / "latest_run.txt").write_text(run_name, encoding="utf-8")
        elapsed = time.monotonic() - started
        pure_training_seconds = max(
            elapsed - validation_callback.excluded_validation_seconds,
            0.0,
        )
        metrics = {
            "run_name": run_name,
            "total_timesteps": total_steps,
            "training_seconds": elapsed,
            "pure_training_seconds": pure_training_seconds,
            "validation_seconds": validation_callback.excluded_validation_seconds,
            "steps_per_second": int(model.num_timesteps) / max(elapsed, 1.0e-9),
            "steps_completed": int(model.num_timesteps),
            "stopped_early": validation_callback.stopped_early,
            "stop_step": validation_callback.stop_step,
            "policy_best": (str(run_directory / "policy_best.zip")
                            if (run_directory / "policy_best.zip").is_file() else None),
            "gradient_updates": int(model._n_updates),
            "policy_final": str(run_directory / "policy_final.zip"),
        }
        (run_directory / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if (run_directory / "policy_best.zip").is_file():
            print(f"Saved: {run_directory / 'policy_best.zip'}")
        print(f"Saved: {run_directory / 'policy_final.zip'}")
        print(f"Training time: {_format_seconds(elapsed)}")
    finally:
        environment.close()
        evaluation_environment.close()


if __name__ == "__main__":
    main()
