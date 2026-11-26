"""정책 학습 (Stable-Baselines3 SAC 기본, PPO 선택).

예) python -m pickplace.train --exp config/exp_A_base.yaml --run A_base_s7
    python -m pickplace.train --exp config/exp_A_sym.yaml  --run A_sym_s7
    python -m pickplace.train --exp config/exp_B_dr.yaml   --run B_dr_s7
    빠른 동작 확인: --steps 20000

산출물 runs/<run>/ : run_config.yaml, policy_final.zip, policy_best.zip, checkpoints/,
                    monitor/*.csv(학습 곡선), eval_history.csv, tensorboard/
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .env import ROOT, PickPlaceEnv, load_config


def make_env_fn(cfg: dict, rank: int, seed: int, monitor_dir: Path):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = PickPlaceEnv(config=cfg)
        env.reset(seed=seed + rank)
        return Monitor(env, filename=str(monitor_dir / f"env{rank}"),
                       info_keywords=("is_success", "end_reason", "max_stage"))
    return _init


def quick_eval(model, cfg: dict, episodes: int, seed: int) -> dict:
    """학습 중 검증: 고정 seed 집합(공식 Seed 0~9와 겹치지 않음), 전체 범위·결정적 행동."""
    env = PickPlaceEnv(config=cfg)
    succ, stages = 0, []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
        succ += info["is_success"]
        stages.append(info["max_stage"])
    env.close()
    return {"success_rate": succ / episodes, "mean_max_stage": float(np.mean(stages))}


def collect_demo_transitions(
    cfg: dict, n_episodes: int, seed_start: int = 20000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """스크립트 성공 시연 궤적을 수집한다(시드 20000+, tanh 경계 포화 방지 위해 행동 ±0.85 클립)."""
    from tools.scripted_check import scripted_action

    env = PickPlaceEnv(config=cfg)
    env.set_curriculum(1.0)
    obs_l, next_l, act_l, rew_l, done_l = [], [], [], [], []
    ok_eps = 0
    last_s = seed_start
    max_attempts = max(n_episodes * 5, 100)
    for s in range(seed_start, seed_start + max_attempts):
        last_s = s
        obs, _ = env.reset(seed=s)
        phase = {"name": "above", "t": 0}
        ep: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
        done = False
        info: dict = {}
        while not done:
            raw_act = scripted_action(env, phase).astype(np.float32)
            act = np.clip(raw_act, -0.85, 0.85).astype(np.float32)
            next_obs, rew, term, trunc, info = env.step(act)
            done = bool(term or trunc)
            ep.append((obs.astype(np.float32), next_obs.astype(np.float32), act, float(rew), float(done)))
            obs = next_obs
        if info.get("is_success"):
            for o, no, a, r, d in ep:
                obs_l.append(o); next_l.append(no); act_l.append(a); rew_l.append(r); done_l.append(d)
            ok_eps += 1
            if ok_eps >= n_episodes:
                break
    env.close()
    print(
        f"[demo] collected {len(obs_l)} transitions from {ok_eps}/{n_episodes} successful episodes "
        f"(seeds {seed_start}..{last_s})",
        flush=True,
    )
    return (
        np.asarray(obs_l, dtype=np.float32),
        np.asarray(next_l, dtype=np.float32),
        np.asarray(act_l, dtype=np.float32),
        np.asarray(rew_l, dtype=np.float32),
        np.asarray(done_l, dtype=np.float32),
        ok_eps,
        last_s,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=[], help="덮어쓸 실험 설정 yaml (순서대로)")
    ap.add_argument("--run", required=True, help="runs/ 아래 폴더 이름")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--demo-episodes", type=int, default=None, help="RLPD 시연 혼합 버퍼용 성공 시연 수")
    ap.add_argument("--resume", default=None, help="이어 학습할 policy zip")
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from .rl import DemoMixReplayBuffer, LayerNormSACPolicy, StableSAC

    cfg = load_config(*args.exp)
    tr = cfg["training"]
    steps = int(args.steps or tr["total_timesteps"])
    seed = int(args.seed if args.seed is not None else tr["seed"])
    n_envs = int(args.n_envs or tr["n_envs"])
    demo_episodes = int(args.demo_episodes if args.demo_episodes is not None else tr.get("demo_episodes", 0))
    demo_ratio = float(tr.get("demo_ratio", 0.25))
    demo_anneal_fraction = float(tr.get("demo_anneal_fraction", 0.7))
    torch.set_num_threads(int(tr.get("torch_threads", 2)))

    run_dir = ROOT / "runs" / args.run
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"{run_dir} 가 이미 있습니다. --run 이름을 바꾸세요.")
    (run_dir / "monitor").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    cfg["_effective"] = {"steps": steps, "seed": seed, "n_envs": n_envs, "demo_episodes": demo_episodes, "exp": args.exp}
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                              encoding="utf-8")

    fns = [make_env_fn(cfg, i, seed * 1000 + 100000, run_dir / "monitor") for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    venv.seed(seed)

    algo = tr["algorithm"].lower()
    policy_kwargs = {"net_arch": list(tr["net_arch"])}
    if args.resume:
        model = (StableSAC if algo == "sac" else PPO).load(args.resume, env=venv, device=tr["device"])
    elif algo == "sac":
        rb_cls = DemoMixReplayBuffer if demo_episodes > 0 else None
        model = StableSAC(
            LayerNormSACPolicy, venv, learning_rate=tr["learning_rate"], buffer_size=tr["buffer_size"],
            learning_starts=tr["learning_starts"], batch_size=tr["batch_size"], gamma=tr["gamma"],
            tau=tr["tau"], ent_coef=tr["ent_coef"], train_freq=tr["train_freq"],
            gradient_steps=tr["gradient_steps"], policy_kwargs=policy_kwargs,
            replay_buffer_class=rb_cls, seed=seed, device=tr["device"],
            tensorboard_log=str(run_dir / "tensorboard"), verbose=1,
        )
        if demo_episodes > 0 and isinstance(model.replay_buffer, DemoMixReplayBuffer):
            d_obs, d_next, d_act, d_rew, d_done, ok_eps, _ = collect_demo_transitions(cfg, demo_episodes)
            model.replay_buffer.set_demos(d_obs, d_next, d_act, d_rew, d_done, demo_ratio=demo_ratio)
            print(f"[demo] loaded {len(d_obs)} transitions into DemoMixReplayBuffer ({ok_eps} eps, ratio={demo_ratio})",
                  flush=True)
    elif algo == "ppo":
        p = tr["ppo"]
        model = PPO("MlpPolicy", venv, learning_rate=tr["learning_rate"], n_steps=p["n_steps"],
                    batch_size=p["batch_size"], n_epochs=p["n_epochs"], gamma=tr["gamma"],
                    gae_lambda=p["gae_lambda"], clip_range=p["clip_range"], ent_coef=p["ent_coef"],
                    policy_kwargs=policy_kwargs, seed=seed, device=tr["device"],
                    tensorboard_log=str(run_dir / "tensorboard"), verbose=1)
    else:
        raise ValueError(f"unknown algorithm {algo}")

    if isinstance(model, StableSAC):
        model.min_ent_coef = float(tr.get("min_ent_coef", 0.005))
        model.max_ent_coef = float(tr.get("max_ent_coef", 0.20))
        model.max_grad_norm = float(tr.get("max_grad_norm", 10.0))

    cur = cfg["curriculum"]

    class CurriculumAndEval(BaseCallback):
        """커리큘럼(시작 범위 확대) + 시연 혼합비 감쇠 + 주기적 검증·최고 정책 저장."""

        def __init__(self) -> None:
            super().__init__()
            self.next_eval = int(tr["eval_every_steps"])
            self.best_score = (-1.0, -1.0)
            self.t0 = time.time()
            self.hist = run_dir / "eval_history.csv"
            with self.hist.open("w", newline="") as f:
                csv.writer(f).writerow(["timesteps", "curriculum", "success_rate", "mean_max_stage", "elapsed_s"])

        def _scale(self) -> float:
            if not cur["enabled"]:
                return 1.0
            frac = self.num_timesteps / max(1, steps * cur["full_at_fraction"])
            return float(min(1.0, cur["start_scale"] + (1 - cur["start_scale"]) * frac))

        def _demo_ratio(self) -> float:
            if demo_ratio <= 0.0 or demo_anneal_fraction <= 0.0:
                return 0.0
            frac = min(1.0, self.num_timesteps / max(1, steps * demo_anneal_fraction))
            return float(demo_ratio * (1.0 - frac))

        def _on_training_start(self) -> None:
            self.training_env.env_method("set_curriculum", self._scale())
            if hasattr(self.model.replay_buffer, "set_demo_ratio"):
                self.model.replay_buffer.set_demo_ratio(self._demo_ratio())

        def _on_step(self) -> bool:
            if self.n_calls % 1000 == 0:
                self.training_env.env_method("set_curriculum", self._scale())
                if hasattr(self.model.replay_buffer, "set_demo_ratio"):
                    self.model.replay_buffer.set_demo_ratio(self._demo_ratio())
            if self.num_timesteps >= self.next_eval:
                self.next_eval += int(tr["eval_every_steps"])
                r = quick_eval(self.model, cfg, int(tr["eval_episodes"]), seed=50000)
                with self.hist.open("a", newline="") as f:
                    csv.writer(f).writerow([self.num_timesteps, round(self._scale(), 3), r["success_rate"],
                                            r["mean_max_stage"], round(time.time() - self.t0)])
                print(f"[eval] t={self.num_timesteps} curriculum={self._scale():.2f} "
                      f"success={r['success_rate']:.2f} stage={r['mean_max_stage']:.2f}", flush=True)
                score = (float(r["success_rate"]), float(r["mean_max_stage"]))
                if score > self.best_score:
                    self.best_score = score
                    self.model.save(str(run_dir / "policy_best"))
            return True

    callbacks = CallbackList([
        CurriculumAndEval(),
        CheckpointCallback(save_freq=max(1, int(tr["checkpoint_every_steps"]) // n_envs),
                           save_path=str(run_dir / "checkpoints"), name_prefix="ckpt"),
    ])
    t0 = time.time()
    try:
        model.learn(total_timesteps=steps, callback=callbacks, tb_log_name=algo,
                    reset_num_timesteps=not args.resume)
    finally:
        model.save(str(run_dir / "policy_final"))
        (run_dir / "metrics.json").write_text(json.dumps({
            "run": args.run, "algorithm": algo, "steps": int(model.num_timesteps),
            "train_seconds": round(time.time() - t0, 1), "n_envs": n_envs, "seed": seed,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (ROOT / "runs" / "latest_run.txt").write_text(args.run, encoding="utf-8")
        venv.close()
        print(f"saved {run_dir / 'policy_final.zip'}")


if __name__ == "__main__":
    main()
