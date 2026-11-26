"""정책 평가 — 공식 Seed 0~9 + 추가 무작위 회차, yaw 구간별 성공률, 놓은 위치 여유.

예) python -m pickplace.evaluate --run A_sym_s7                       # runs/A_sym_s7/policy_best.zip
    python -m pickplace.evaluate --run A_sym_s7 --env-exp config/eval_noise.yaml --tag noise
    python -m pickplace.evaluate --policy path/to/policy.zip --exp config/exp_A_sym.yaml

산출물 runs/<run>/eval_<tag>/ : episodes.csv (회차별 배치·성공·종료 원인·여유), summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from .env import ROOT, PickPlaceEnv, deep_update, load_config


def load_policy(path: Path, algo: str):
    from stable_baselines3 import PPO, SAC
    return (SAC if algo == "sac" else PPO).load(str(path), device="cpu")


def run_episode(model, env: PickPlaceEnv, seed: int, frame_cb=None) -> dict:
    obs, info = env.reset(seed=seed)
    done = False
    if frame_cb:
        frame_cb(env, info)
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
        if frame_cb:
            frame_cb(env, info)
    return {
        "seed": seed, "start_x": info["start_xy"][0], "start_y": info["start_xy"][1],
        "start_yaw_deg": float(np.degrees(info["start_yaw"])), "success": bool(info["is_success"]),
        "end_reason": info["end_reason"], "max_stage": info["max_stage"], "drops": info["drops"],
        "sim_time_s": round(info["sim_time"], 2), "final_margin_mm": round(info["goal_margin_m"] * 1000, 2),
        "final_yaw_deg": float(np.degrees(info["cube_yaw"])),
        "events": ";".join(f"{t}:{n}" for t, n in env.tracker.events),
    }


def summarize(rows: list[dict], bins: int) -> dict:
    succ = np.array([r["success"] for r in rows])
    oneshot = np.array([bool(r["success"]) and int(r["drops"]) == 0 for r in rows])
    yaw = np.array([r["start_yaw_deg"] for r in rows])
    edges = np.linspace(-180, 180, bins + 1)
    per_bin = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (yaw >= lo) & (yaw < hi)
        per_bin.append({"bin": f"[{lo:.0f},{hi:.0f})", "n": int(m.sum()),
                        "success_rate": float(succ[m].mean()) if m.any() else None})
    reasons = {}
    for r in rows:
        reasons[r["end_reason"]] = reasons.get(r["end_reason"], 0) + 1
    margins = [r["final_margin_mm"] for r in rows if r["success"]]
    times = [r["sim_time_s"] for r in rows if r["success"]]
    return {
        "episodes": len(rows), "successes": int(succ.sum()), "success_rate": float(succ.mean()),
        "oneshot_successes": int(oneshot.sum()), "oneshot_success_rate": float(oneshot.mean()),
        "end_reasons": reasons, "yaw_bins": per_bin,
        "success_margin_mm_mean": float(np.mean(margins)) if margins else None,
        "success_margin_mm_min": float(np.min(margins)) if margins else None,
        "success_time_s_mean": float(np.mean(times)) if times else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--exp", nargs="*", default=[], help="--policy 사용 시 학습 설정")
    ap.add_argument("--env-exp", nargs="*", default=[], help="평가 환경만 바꿀 설정(예: eval_noise.yaml)")
    ap.add_argument("--tag", default="nominal")
    ap.add_argument("--extra", type=int, default=None, help="추가 무작위 회차 수")
    args = ap.parse_args()

    if args.run:
        run_dir = ROOT / "runs" / args.run
        cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
        policy = Path(args.policy) if args.policy else run_dir / f"policy_{args.which}.zip"
    else:
        cfg = load_config(*args.exp)
        policy = Path(args.policy)
        run_dir = policy.parent
    for p in args.env_exp:
        cfg = deep_update(cfg, yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {})
    model = load_policy(policy, cfg["training"]["algorithm"])
    env = PickPlaceEnv(config=cfg)
    ev = cfg["evaluation"]
    n_extra = ev["extra_episodes"] if args.extra is None else args.extra
    out = run_dir / f"eval_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    official = [run_episode(model, env, s) for s in ev["official_seeds"]]
    extra = [run_episode(model, env, ev["extra_seed_start"] + i) for i in range(n_extra)]
    for name, rows in (("official", official), ("extra", extra)):
        if not rows:
            continue
        with (out / f"episodes_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summary = {"policy": str(policy), "tag": args.tag,
               "official": summarize(official, ev["yaw_bins"]),
               "extra": summarize(extra, ev["yaw_bins"]) if extra else None,
               "all": summarize(official + extra, ev["yaw_bins"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Seed  start(x,y,yaw)              result               t[s]  margin[mm]")
    for r in official:
        print(f"{r['seed']:>4}  ({r['start_x']:.3f},{r['start_y']:+.3f},{r['start_yaw_deg']:+7.1f}°)  "
              f"{r['end_reason']:<20} {r['sim_time_s']:5.1f}  {r['final_margin_mm']:+6.1f}")
    o = summary["official"]
    print(f"official {o['successes']}/{o['episodes']} (oneshot {o['oneshot_successes']}/{o['episodes']})  "
          f"end_reasons={o['end_reasons']}")
    if extra:
        e = summary["extra"]
        print(f"extra {e['successes']}/{e['episodes']} ({e['success_rate']*100:.1f}%, "
              f"oneshot {e['oneshot_successes']}/{e['episodes']}={e['oneshot_success_rate']*100:.1f}%)  "
              f"end_reasons={e['end_reasons']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
