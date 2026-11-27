"""정책 학습 (Stable-Baselines3 PPO 기본 + BC 워밍업 / SAC 선택).

예) python -m pickplace.train --exp config/exp_A_base.yaml --run A_base_v4
    python -m pickplace.train --exp config/exp_A_sym.yaml  --run A_sym_v4
    python -m pickplace.train --exp config/exp_B_dr.yaml   --run B_dr_v4
    빠른 동작 확인: --steps 16384

산출물 runs/<run>/ : run_config.yaml, policy_final.zip, policy_best.zip, checkpoints/,
                    monitor/*.csv(학습 곡선), eval_history.csv, tensorboard/
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .env import ROOT, PickPlaceEnv, load_config
from .geometry import align_error, bottom_corners, fold_quarter, region_margin
from .kinematics import GRIPPER_INDEX


def make_env_fn(cfg: dict, rank: int, seed: int, monitor_dir: Path):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = PickPlaceEnv(config=cfg)
        env.reset(seed=seed + rank)
        return Monitor(env, filename=str(monitor_dir / f"env{rank}"),
                       info_keywords=("is_success", "end_reason", "max_stage"))
    return _init


def quick_eval(model, cfg: dict, episodes: int, seed: int, vec_norm=None) -> dict:
    """학습 중 검증: 고정 seed 집합(공식 Seed 0~9와 겹치지 않음), 전체 범위·결정적 행동."""
    env = PickPlaceEnv(config=cfg)
    succ, stages = 0, []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            obs_in = vec_norm.normalize_obs(obs) if (vec_norm is not None and vec_norm.norm_obs) else obs
            act, _ = model.predict(obs_in, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
        succ += info["is_success"]
        stages.append(info["max_stage"])
    env.close()
    return {"success_rate": succ / episodes, "mean_max_stage": float(np.mean(stages))}


CARRY_Z = 0.036


def markov_demo_action(env: PickPlaceEnv, state: dict | None = None) -> np.ndarray:
    """오프셋 보상·선행 Yaw 정렬·바닥 3스텝 안착 후 무드리프트 해제·목표 밖 복구 적용 시연 전문가."""
    if state is None:
        state = {"carry": False, "releasing": False, "settled": 0}
    q = env.joint_q()
    f = env.kin.fk(q)
    gp, gyaw = f["grasp_point"], f["gripper_yaw"]
    cube = env.cube_pos()
    crot = env.cube_rot()
    corners = bottom_corners(cube, crot, env.half)
    margin = region_margin(corners, env.goal_xy, env.goal_half)
    a = np.zeros(5, dtype=np.float32)

    def move_to(target: np.ndarray, gain: float = 1.0) -> np.ndarray:
        return np.clip(gain * (target - gp) / env.act_scale[:3], -1.0, 1.0)

    e = align_error(env.cube_yaw(), gyaw)
    yaw45 = fold_quarter(env.cube_yaw())
    grasp_pt = cube + np.array([0.0, 0.0, 0.001])
    d_gp_cube_xy = float(np.linalg.norm(gp[:2] - grasp_pt[:2]))
    d_gp_cube = float(np.linalg.norm(gp - grasp_pt))
    gq = float(q[GRIPPER_INDEX])

    if env.tracker.stage == 3 and margin >= 0.004:
        state["settled"] = state.get("settled", 0) + 1
    else:
        state["settled"] = 0

    if state.get("releasing", False) or state["settled"] >= 3:
        if gq >= 0.50 and margin < 0.0:
            state["releasing"] = False
            state["carry"] = False
            state["settled"] = 0
        else:
            state["carry"] = False
            state["releasing"] = True
            if gq < 0.50:
                a[:4] = 0.0
                a[4] = -1.0
            else:
                a[:3] = move_to(np.array([gp[0], gp[1], 0.08]))
                a[4] = -1.0
            return a

    gripped = env.grip_closed_cmd and (-0.06 < gq < 0.10)
    if not gripped:
        state["carry"] = False
    if gripped:
        a[4] = 1.0
        grasp_off = np.clip(cube[:2] - gp[:2], -0.015, 0.015)
        place_xy = env.goal_xy - grasp_off
        place_xy[0] = min(place_xy[0], 0.393)
        d_target_xy = float(np.linalg.norm(gp[:2] - place_xy))
        if d_target_xy > 0.010 or abs(yaw45) > 0.14:
            if gp[2] > CARRY_Z - 0.005:
                state["carry"] = True
            if not state["carry"]:
                a[:3] = move_to(np.array([gp[0], gp[1], CARRY_Z]))
            else:
                a[:3] = move_to(np.array([place_xy[0], place_xy[1], CARRY_Z]))
                a[3] = float(np.clip(-0.7 * yaw45 / env.act_scale[3], -1.0, 1.0))
        else:
            a[:3] = move_to(np.array([place_xy[0], place_xy[1], env.rest_z - 0.003]))
            a[3] = float(np.clip(-0.7 * yaw45 / env.act_scale[3], -1.0, 1.0))
        return a

    if d_gp_cube < 0.007 and abs(e) < 0.08:
        a[:3] = move_to(grasp_pt, gain=0.5)
        a[3] = float(np.clip(0.5 * e / env.act_scale[3], -1.0, 1.0))
        a[4] = 1.0
    elif d_gp_cube_xy < 0.010 and abs(e) < 0.08:
        a[:3] = move_to(grasp_pt)
        a[3] = float(np.clip(0.5 * e / env.act_scale[3], -1.0, 1.0))
        a[4] = -1.0
    else:
        a[:3] = move_to(cube + np.array([0.0, 0.0, 0.05]))
        a[3] = float(np.clip(0.5 * e / env.act_scale[3], -1.0, 1.0))
        a[4] = -1.0
    return a


def collect_demo_transitions(
    cfg: dict, n_episodes: int, seed_start: int = 20000, markov: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """성공 시연 궤적 및 할인수익 수집(시드 20000+, DR 시 관측 오염은 제거해 물리 DR만 유지)."""
    from tools.scripted_check import scripted_action

    gamma = float(cfg["training"]["gamma"])
    demo_cfg = copy.deepcopy(cfg)
    if demo_cfg.get("randomization", {}).get("enabled", False):
        dr = demo_cfg["randomization"]
        dr["obs_pos_bias_m"] = 0.0
        dr["obs_pos_noise_m"] = 0.0
        dr["obs_yaw_bias_rad"] = 0.0
        dr["obs_yaw_noise_rad"] = 0.0
        dr["obs_delay_steps"] = [0, 0]
        dr["obs_dropout_prob"] = 0.0
        dr["occlusion_radius_m"] = 0.0
    env = PickPlaceEnv(config=demo_cfg)
    env.set_curriculum(1.0)
    obs_l, next_l, act_l, rew_l, done_l, ret_l = [], [], [], [], [], []
    ok_eps = 0
    last_s = seed_start
    max_attempts = max(n_episodes * 5, 100)
    for s in range(seed_start, seed_start + max_attempts):
        last_s = s
        obs, _ = env.reset(seed=s)
        phase = {"name": "above", "t": 0}
        expert_state = {"carry": False, "releasing": False, "settled": 0}
        ep: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
        done = False
        info: dict = {}
        ep_steps = 0
        while not done and (not markov or ep_steps < 280):
            raw_act = (markov_demo_action(env, expert_state) if markov else scripted_action(env, phase)).astype(np.float32)
            act = np.clip(raw_act, -1.0 if markov else -0.85, 1.0 if markov else 0.85).astype(np.float32)
            next_obs, rew, term, trunc, info = env.step(act)
            done = bool(term or trunc)
            ep.append((obs.astype(np.float32), next_obs.astype(np.float32), act, float(rew), float(done)))
            obs = next_obs
            ep_steps += 1
        if info.get("is_success"):
            rets = np.zeros(len(ep), dtype=np.float32)
            g = 0.0
            for idx in range(len(ep) - 1, -1, -1):
                g = ep[idx][3] + gamma * g
                rets[idx] = g
            for (o, no, a, r, d), ret_val in zip(ep, rets):
                obs_l.append(o); next_l.append(no); act_l.append(a)
                rew_l.append(r); done_l.append(d); ret_l.append(float(ret_val))
            ok_eps += 1
            if ok_eps >= n_episodes:
                break
    env.close()
    print(
        f"[demo] collected {len(obs_l)} transitions from {ok_eps}/{n_episodes} successful episodes "
        f"(seeds {seed_start}..{last_s}, markov={markov})",
        flush=True,
    )
    return (
        np.asarray(obs_l, dtype=np.float32),
        np.asarray(next_l, dtype=np.float32),
        np.asarray(act_l, dtype=np.float32),
        np.asarray(rew_l, dtype=np.float32),
        np.asarray(done_l, dtype=np.float32),
        np.asarray(ret_l, dtype=np.float32),
        ok_eps,
        last_s,
    )


def save_folded_ppo(model, vec_norm, path: Path) -> None:
    """VecNormalize 통계를 PPO 1층 가중치·편향에 병합해 표준 PPO.load() 호환 파일로 저장한다."""
    import torch

    if vec_norm is None or not vec_norm.norm_obs:
        model.save(str(path))
        return

    orig_state = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    mean = torch.as_tensor(vec_norm.obs_rms.mean, dtype=torch.float32, device=model.device)
    var = torch.as_tensor(vec_norm.obs_rms.var, dtype=torch.float32, device=model.device)
    inv_std = 1.0 / torch.sqrt(var + float(vec_norm.epsilon))

    try:
        with torch.no_grad():
            for layer in (model.policy.mlp_extractor.policy_net[0], model.policy.mlp_extractor.value_net[0]):
                w_new = layer.weight * inv_std.unsqueeze(0)
                b_new = layer.bias - (w_new @ mean)
                layer.weight.copy_(w_new)
                layer.bias.copy_(b_new)
        model.save(str(path))
    finally:
        model.policy.load_state_dict(orig_state)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=[], help="덮어쓸 실험 설정 yaml (순서대로)")
    ap.add_argument("--run", required=True, help="runs/ 아래 폴더 이름")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--demo-episodes", type=int, default=None, help="시연 워밍업/버퍼용 성공 시연 수")
    ap.add_argument("--drop-penalty", type=float, default=None, help="운반 중 낙하 1회당 감점 덮어쓰기")
    ap.add_argument("--resume", default=None, help="이어 학습할 policy zip")
    args = ap.parse_args()

    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from .rl import AnchoredPPO, DemoMixReplayBuffer, LayerNormSACPolicy, StableSAC

    cfg = load_config(*args.exp)
    if args.drop_penalty is not None:
        cfg["reward"]["drop_penalty"] = float(args.drop_penalty)
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
    base_venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    base_venv.seed(seed)

    algo = tr["algorithm"].lower()
    vec_norm = None
    train_env = base_venv

    if algo == "sac":
        policy_kwargs = {"net_arch": list(tr["net_arch"])}
        if args.resume:
            model = StableSAC.load(args.resume, env=train_env, device=tr["device"])
        else:
            rb_cls = DemoMixReplayBuffer if demo_episodes > 0 else None
            model = StableSAC(
                LayerNormSACPolicy, train_env, learning_rate=tr["learning_rate"], buffer_size=tr["buffer_size"],
                learning_starts=tr["learning_starts"], batch_size=tr["batch_size"], gamma=tr["gamma"],
                tau=tr["tau"], ent_coef=tr["ent_coef"], train_freq=tr["train_freq"],
                gradient_steps=tr["gradient_steps"], policy_kwargs=policy_kwargs,
                replay_buffer_class=rb_cls, seed=seed, device=tr["device"],
                tensorboard_log=str(run_dir / "tensorboard"), verbose=1,
            )
            if demo_episodes > 0 and isinstance(model.replay_buffer, DemoMixReplayBuffer):
                d_obs, d_next, d_act, d_rew, d_done, _, ok_eps, _ = collect_demo_transitions(
                    cfg, demo_episodes, markov=False
                )
                model.replay_buffer.set_demos(d_obs, d_next, d_act, d_rew, d_done, demo_ratio=demo_ratio)
                print(f"[demo] loaded {len(d_obs)} transitions into DemoMixReplayBuffer ({ok_eps} eps, ratio={demo_ratio})",
                      flush=True)
    elif algo == "ppo":
        p = tr["ppo"]
        vec_norm = VecNormalize(
            base_venv, norm_obs=not bool(args.resume), norm_reward=True,
            clip_obs=100.0, clip_reward=10.0, gamma=float(tr["gamma"]),
        )
        train_env = vec_norm
        lr_start = float(tr["learning_rate"])
        lr_end = lr_start * 0.2
        lin_lr = lambda rem: float(lr_end + (lr_start - lr_end) * rem)  # noqa: E731
        policy_kwargs = {"net_arch": dict(pi=list(tr["net_arch"]), vf=list(tr["net_arch"]))}
        if args.resume:
            model = AnchoredPPO.load(args.resume, env=train_env, device=tr["device"])
        else:
            model = AnchoredPPO(
                "MlpPolicy", train_env, learning_rate=lin_lr, n_steps=int(p["n_steps"]),
                batch_size=int(p["batch_size"]), n_epochs=int(p["n_epochs"]), gamma=float(tr["gamma"]),
                gae_lambda=float(p["gae_lambda"]), clip_range=float(p["clip_range"]),
                target_kl=float(p.get("target_kl", 0.015)), ent_coef=float(p["ent_coef"]),
                vf_coef=float(p.get("vf_coef", 0.5)), max_grad_norm=float(p.get("max_grad_norm", 0.5)),
                policy_kwargs=policy_kwargs, seed=seed, device=tr["device"],
                tensorboard_log=str(run_dir / "tensorboard"), verbose=1,
            )
            if demo_episodes > 0:
                d_obs, _, d_act, _, _, d_ret, ok_eps, _ = collect_demo_transitions(
                    cfg, demo_episodes, markov=True
                )
                vec_norm.obs_rms.mean = d_obs.mean(axis=0).astype(np.float64)
                vec_norm.obs_rms.var = np.maximum(d_obs.var(axis=0).astype(np.float64), 1e-6)
                vec_norm.obs_rms.count = 1e9
                vec_norm.ret_rms.mean = float(d_ret.mean())
                vec_norm.ret_rms.var = max(float(d_ret.var()), 1.0)
                vec_norm.ret_rms.count = float(len(d_ret))

                obs_mean = torch.as_tensor(vec_norm.obs_rms.mean, dtype=torch.float32, device=model.device)
                obs_std = torch.sqrt(
                    torch.as_tensor(vec_norm.obs_rms.var, dtype=torch.float32, device=model.device) + float(vec_norm.epsilon)
                )
                ret_std = float(np.sqrt(vec_norm.ret_rms.var + float(vec_norm.epsilon)))

                X = torch.as_tensor(d_obs, dtype=torch.float32, device=model.device)
                Y = torch.as_tensor(d_act, dtype=torch.float32, device=model.device)
                G = torch.as_tensor(d_ret / ret_std, dtype=torch.float32, device=model.device)
                Xn = torch.clamp((X - obs_mean) / obs_std, -float(vec_norm.clip_obs), float(vec_norm.clip_obs))

                bc_epochs = int(tr.get("bc_epochs", 120))
                bc_lr = float(tr.get("bc_lr", 0.003))
                bc_bs = 256
                opt = torch.optim.Adam(model.policy.parameters(), lr=bc_lr)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, bc_epochs), eta_min=bc_lr * 0.1)
                model.policy.set_training_mode(True)
                last_actor_mse = last_critic_mse = 0.0
                for _ in range(bc_epochs):
                    idx = torch.randperm(len(Xn), device=model.device)
                    for i in range(0, len(Xn), bc_bs):
                        b = idx[i : i + bc_bs]
                        pred_act = model.policy.get_distribution(Xn[b]).mode()
                        pred_val = model.policy.predict_values(Xn[b]).squeeze(-1)
                        actor_mse = ((pred_act - Y[b]) ** 2).mean()
                        critic_mse = ((pred_val - G[b]) ** 2).mean()
                        loss = actor_mse + 0.5 * critic_mse
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
                        opt.step()
                        last_actor_mse = float(actor_mse.item())
                        last_critic_mse = float(critic_mse.item())
                    sched.step()
                with torch.no_grad():
                    model.policy.log_std.data.fill_(float(tr.get("bc_init_log_std", -2.6)))
                print(
                    f"[bc] epochs={bc_epochs} eps={ok_eps} steps={len(d_obs)} "
                    f"actor_mse={last_actor_mse:.5f} critic_mse={last_critic_mse:.5f}",
                    flush=True,
                )

                X_all, Y_all = X, Y
                dagger_rounds = int(tr.get("dagger_rounds", 0))
                dagger_episodes = int(tr.get("dagger_episodes", 30))
                dagger_epochs = int(tr.get("dagger_epochs", 60))
                if dagger_rounds > 0 and dagger_episodes > 0:
                    d_env = PickPlaceEnv(config=cfg)
                    d_env.set_curriculum(1.0)
                    actor_params = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
                        model.policy.action_net.parameters()
                    )
                    for r_idx in range(dagger_rounds):
                        new_obs, new_act = [], []
                        n_ok = 0
                        for ep_i in range(dagger_episodes):
                            o, _ = d_env.reset(seed=21000 + r_idx * 1000 + ep_i)
                            expert_state = {"carry": False, "releasing": False, "settled": 0}
                            done = False
                            ep_st = 0
                            info_d: dict = {}
                            while not done and ep_st < 280:
                                expert_a = markov_demo_action(d_env, expert_state)
                                new_obs.append(o.astype(np.float32))
                                new_act.append(np.clip(expert_a, -1.0, 1.0).astype(np.float32))
                                o_in = vec_norm.normalize_obs(o)
                                act_p, _ = model.predict(o_in, deterministic=True)
                                o, _, term, trunc, info_d = d_env.step(act_p)
                                done = term or trunc
                                ep_st += 1
                            n_ok += int(info_d.get("is_success", False))
                        X_new = torch.as_tensor(np.asarray(new_obs, dtype=np.float32), device=model.device)
                        Y_new = torch.as_tensor(np.asarray(new_act, dtype=np.float32), device=model.device)
                        X_all = torch.cat([X_all, X_new], dim=0)
                        Y_all = torch.cat([Y_all, Y_new], dim=0)
                        Xn_all = torch.clamp(
                            (X_all - obs_mean) / obs_std, -float(vec_norm.clip_obs), float(vec_norm.clip_obs)
                        )
                        opt_d = torch.optim.Adam(actor_params, lr=0.002)
                        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                            opt_d, T_max=max(1, dagger_epochs), eta_min=0.0002
                        )
                        model.policy.set_training_mode(True)
                        last_d_mse = 0.0
                        for _ in range(dagger_epochs):
                            idx_d = torch.randperm(len(Xn_all), device=model.device)
                            for i in range(0, len(Xn_all), bc_bs):
                                b = idx_d[i : i + bc_bs]
                                pred_act = model.policy.get_distribution(Xn_all[b]).mode()
                                d_loss = ((pred_act - Y_all[b]) ** 2).mean()
                                opt_d.zero_grad()
                                d_loss.backward()
                                torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
                                opt_d.step()
                                last_d_mse = float(d_loss.item())
                            sched_d.step()
                        with torch.no_grad():
                            model.policy.log_std.data.fill_(float(tr.get("bc_init_log_std", -2.6)))
                        print(
                            f"[dagger r={r_idx + 1}] rollout_succ={n_ok}/{dagger_episodes} "
                            f"added={len(new_obs)} total={len(X_all)} actor_mse={last_d_mse:.5f}",
                            flush=True,
                        )
                    d_env.close()
                Xn_anchor = torch.clamp(
                    (X_all - obs_mean) / obs_std, -float(vec_norm.clip_obs), float(vec_norm.clip_obs)
                )
                model.set_bc_anchor(
                    Xn_anchor, Y_all,
                    coef_start=float(p.get("bc_anchor_start", 0.35)),
                    coef_end=float(p.get("bc_anchor_end", 0.05)),
                )
        model.adv_neg_scale = float(p.get("adv_neg_scale", 0.25))
        model.min_std = float(p.get("min_std", 0.03))
        model.max_std = float(p.get("max_std", 0.10))
    else:
        raise ValueError(f"unknown algorithm {algo}")

    if isinstance(model, StableSAC):
        model.min_ent_coef = float(tr.get("min_ent_coef", 0.005))
        model.max_ent_coef = float(tr.get("max_ent_coef", 0.20))
        model.max_grad_norm = float(tr.get("max_grad_norm", 10.0))

    cur = cfg["curriculum"]
    vf_warmup_iters = int(tr.get("vf_warmup_iters", 0)) if (algo == "ppo" and not args.resume and demo_episodes > 0) else 0

    class CurriculumAndEval(BaseCallback):
        """커리큘럼 + Critic 워밍업 + 주기적 검증·최고 정책 저장."""

        def __init__(self) -> None:
            super().__init__()
            self.next_eval = int(tr["eval_every_steps"])
            self.last_eval_t = -1
            self.best_score = (-1.0, -1.0)
            self.t0 = time.time()
            self.n_rollouts = 0
            self.actor_frozen = False
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

        def set_actor_grad(self, enabled: bool) -> None:
            for mod in (self.model.policy.mlp_extractor.policy_net, self.model.policy.action_net):
                for p in mod.parameters():
                    p.requires_grad = enabled
            self.model.policy.log_std.requires_grad = enabled
            self.actor_frozen = not enabled

        def _save_policy(self, path: Path) -> None:
            if vec_norm is not None:
                save_folded_ppo(self.model, vec_norm, path)
            else:
                self.model.save(str(path))

        def _run_eval_and_log(self, t_val: int) -> None:
            self.last_eval_t = int(t_val)
            r = quick_eval(self.model, cfg, int(tr["eval_episodes"]), seed=50000, vec_norm=vec_norm)
            with self.hist.open("a", newline="") as f:
                csv.writer(f).writerow([t_val, round(self._scale(), 3), r["success_rate"],
                                        r["mean_max_stage"], round(time.time() - self.t0)])
            print(f"[eval] t={t_val} curriculum={self._scale():.2f} "
                  f"success={r['success_rate']:.2f} stage={r['mean_max_stage']:.2f}", flush=True)
            score = (float(r["success_rate"]), float(r["mean_max_stage"]))
            if score > self.best_score:
                self.best_score = score
                self._save_policy(run_dir / "policy_best")

        def _on_training_start(self) -> None:
            base_venv.env_method("set_curriculum", self._scale())
            rb = getattr(self.model, "replay_buffer", None)
            if rb is not None and hasattr(rb, "set_demo_ratio"):
                rb.set_demo_ratio(self._demo_ratio())
            self._run_eval_and_log(self.num_timesteps)

        def _on_rollout_end(self) -> None:
            if vf_warmup_iters > 0:
                if self.n_rollouts < vf_warmup_iters:
                    self.set_actor_grad(False)
                elif self.actor_frozen:
                    self.set_actor_grad(True)
                    print(f"[ppo] critic warmup done (iter={self.n_rollouts}), unfreezing actor", flush=True)
            self.n_rollouts += 1

        def _on_step(self) -> bool:
            if self.n_calls % 1000 == 0:
                base_venv.env_method("set_curriculum", self._scale())
                rb = getattr(self.model, "replay_buffer", None)
                if rb is not None and hasattr(rb, "set_demo_ratio"):
                    rb.set_demo_ratio(self._demo_ratio())
            if self.num_timesteps >= self.next_eval:
                while self.num_timesteps >= self.next_eval:
                    self.next_eval += int(tr["eval_every_steps"])
                self._run_eval_and_log(self.num_timesteps)
            return True

    eval_cb = CurriculumAndEval()
    callbacks = CallbackList([
        eval_cb,
        CheckpointCallback(save_freq=max(1, int(tr["checkpoint_every_steps"]) // n_envs),
                           save_path=str(run_dir / "checkpoints"), name_prefix="ckpt"),
    ])
    t0 = time.time()
    try:
        model.learn(total_timesteps=steps, callback=callbacks, tb_log_name=algo,
                    reset_num_timesteps=not args.resume)
        if eval_cb.num_timesteps > eval_cb.last_eval_t:
            eval_cb._run_eval_and_log(eval_cb.num_timesteps)
    finally:
        if algo == "ppo" and eval_cb.actor_frozen:
            eval_cb.set_actor_grad(True)
        if vec_norm is not None:
            save_folded_ppo(model, vec_norm, run_dir / "policy_final")
        else:
            model.save(str(run_dir / "policy_final"))
        (run_dir / "metrics.json").write_text(json.dumps({
            "run": args.run, "algorithm": algo, "steps": int(model.num_timesteps),
            "train_seconds": round(time.time() - t0, 1), "n_envs": n_envs, "seed": seed,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (ROOT / "runs" / "latest_run.txt").write_text(args.run, encoding="utf-8")
        train_env.close()
        print(f"saved {run_dir / 'policy_final.zip'}")


if __name__ == "__main__":
    main()
