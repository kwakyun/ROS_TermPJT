"""Evaluate a trained Week 11 policy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .env import PushEnv, project_directory


def resolve_run_directory(run_name: Optional[str] = "latest") -> Path:
    output_root = project_directory() / "outputs"
    if run_name and run_name != "latest":
        run_directory = output_root / run_name
    else:
        latest = output_root / "latest_run.txt"
        if not latest.is_file():
            raise FileNotFoundError("No trained run. Run the train module first.")
        run_directory = output_root / latest.read_text(encoding="utf-8").strip()
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_directory}")
    return run_directory


def resolve_policy_path(
    run_name: Optional[str] = "latest",
    explicit_policy: Optional[str] = None,
    model_name: str = "final",
) -> Path:
    path = (
        Path(explicit_policy).expanduser().resolve()
        if explicit_policy
        else resolve_run_directory(run_name) / f"policy_{model_name}.zip"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Policy archive not found: {path}")
    return path


def evaluate_model(
    *,
    model: Any,
    environment: PushEnv,
    episodes_per_scenario: int,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    scenarios = environment.scenario_names
    for scenario_index, scenario in enumerate(scenarios):
        for episode_index in range(episodes_per_scenario):
            episode_seed = seed + scenario_index * 1000 + episode_index
            observation, info = environment.reset(
                seed=episode_seed, options={"scenario": scenario}
            )
            initial_cube = np.asarray(info["object_position"][:2]).copy()
            previous_eef = np.asarray(info["eef_position"]).copy()
            minimum_distance = float(info["distance_between_cubes"])
            eef_path = 0.0
            contact_steps = 0
            ik_rejected_steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                # TODO 6 — model.predict()에 현재 observation과 deterministic=True를 전달한다.
                # 반환된 (action, state)에서 action을 받아 다음 environment.step()에 사용한다.
                action, _state = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
                current_eef = np.asarray(info["eef_position"])
                eef_path += float(np.linalg.norm(current_eef - previous_eef))
                previous_eef = current_eef
                minimum_distance = min(
                    minimum_distance, float(info["distance_between_cubes"])
                )
                contact_steps += int(bool(info["contact_latched"]))
                ik_rejected_steps += int(bool(info["ik_rejected"]))
            rows.append(
                {
                    "scenario": scenario,
                    "seed": episode_seed,
                    "success": bool(info["is_success"]),
                    "failure_reason": str(info["failure_reason"]),
                    "steps": int(info["elapsed_steps"]),
                    "final_distance_m": float(info["distance_between_cubes"]),
                    "minimum_distance_m": minimum_distance,
                    "eef_path_length_m": eef_path,
                    "ever_contact": contact_steps > 0,
                    "contact_substep": info["contact_substep"],
                    "target_robot_contact": info["target_robot_contact"],
                    "environment_version": environment.environment_version,
                    "ik_rejected_steps": ik_rejected_steps,
                    "cube_displacement_m": float(
                        np.linalg.norm(
                            np.asarray(info["object_position"][:2]) - initial_cube
                        )
                    ),
                }
            )

    summary: Dict[str, Any] = {}
    for scenario in (*scenarios, "overall"):
        subset = rows if scenario == "overall" else [
            row for row in rows if row["scenario"] == scenario
        ]
        successes = [row for row in subset if row["success"]]
        summary[scenario] = {
            "episodes": len(subset),
            "successes": len(successes),
            "success_rate": len(successes) / max(1, len(subset)),
            "mean_steps": float(np.mean([row["steps"] for row in subset])),
            "mean_success_steps": (
                float(np.mean([row["steps"] for row in successes]))
                if successes else None
            ),
            "mean_final_distance_m": float(
                np.mean([row["final_distance_m"] for row in subset])
            ),
            "mean_minimum_distance_m": float(
                np.mean([row["minimum_distance_m"] for row in subset])
            ),
            "mean_eef_path_length_m": float(
                np.mean([row["eef_path_length_m"] for row in subset])
            ),
            "contact_episode_rate": float(
                np.mean([row["ever_contact"] for row in subset])
            ),
            "mean_ik_rejected_steps": float(
                np.mean([row["ik_rejected_steps"] for row in subset])
            ),
        }
    return summary, rows


def print_summary(summary: Dict[str, Any]) -> None:
    for scenario, item in summary.items():
        print(
            f"{scenario:16s}: success={100.0*item['success_rate']:5.1f}% "
            f"({item['successes']}/{item['episodes']}), "
            f"steps={item['mean_steps']:.1f}, "
            f"cube_center_distance={item['mean_final_distance_m']:.4f} m"
        )


def write_evaluation(
    summary: Dict[str, Any], rows: List[Dict[str, Any]], directory: Path
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        with (directory / "evaluation_episodes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    """Load one policy, run deterministic episodes and save the results."""
    from stable_baselines3 import SAC

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="latest")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--model", choices=("best", "final"), default="final")
    parser.add_argument("--episodes-per-scenario", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.episodes_per_scenario is not None and args.episodes_per_scenario <= 0:
        parser.error("--episodes-per-scenario must be positive")
    import torch
    torch.set_num_threads(1)

    with (project_directory() / "config.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        evaluation = (yaml.safe_load(stream) or {})["evaluation"]
    episodes = int(
        args.episodes_per_scenario or evaluation["episodes_per_scenario"]
    )
    seed = int(evaluation["seed"] if args.seed is None else args.seed)
    policy_path = resolve_policy_path(args.run, args.policy, args.model)
    run_directory = (
        policy_path.parent if args.policy else resolve_run_directory(args.run)
    )
    environment = PushEnv()
    try:
        # TODO 5 — SAC.load()로 policy_path의 정책을 불러와 model에 저장한다.
        # 평가 environment를 env에 연결하고 device는 cpu로 지정한다.
        model = SAC.load(str(policy_path), env=environment, device="cpu")
        if getattr(model, "environment_version", None) != environment.environment_version:
            raise ValueError("Policy is from an incompatible environment; retrain for direct_xyz15_distance_v1")
        summary, rows = evaluate_model(
            model=model,
            environment=environment,
            episodes_per_scenario=episodes,
            seed=seed,
        )
        print(f"Policy: {policy_path}")
        print_summary(summary)
        write_evaluation(summary, rows, run_directory / "evaluation_final")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
