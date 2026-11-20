"""Inspect the task and Gymnasium interface before training."""

import numpy as np
from gymnasium.utils.env_checker import check_env

from .env import PushEnv


def main():
    environment = PushEnv()
    try:
        check_env(environment, skip_render_check=True)
        observation, info = environment.reset(seed=12)
        print("observation:", environment.observation_space)
        print("action:", environment.action_space)
        for key in ("eef_position", "moving_tip_position", "object_position", "target_position"):
            print(key, np.round(info[key], 6))
        assert observation.shape == (15,) and observation.dtype == np.float32
        action = np.array([0.5, -0.25, 1.0], dtype=np.float32)
        before = info["eef_position"].copy()
        observation, reward, terminated, truncated, info = environment.step(action)
        print("action XYZ:", action)
        print("requested XYZ (m):", action * np.array([0.008, 0.008, 0.005]))
        print("measured XYZ (m):", np.round(info["eef_position"] - before, 6))
        print("reward terms:", info["reward_terms"])
        print("reward:", reward)
        print("terminated/truncated:", terminated, truncated)
        print("Gymnasium API OK")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
