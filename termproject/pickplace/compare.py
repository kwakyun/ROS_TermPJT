"""비교실험 그래프·표 생성 (보고서 '비교실험' 10점용).

예) 비교 A: python -m pickplace.compare --runs A_base_s7 A_sym_s7 --labels base sym --tag nominal --out reports/compare_A
    비교 B: python -m pickplace.compare --runs A_sym_s7 B_dr_s7 --labels noDR DR --tag noise --out reports/compare_B
    (학습 seed 여러 개면 --runs A_base_s7 A_base_s8 ... --labels base base ... 처럼 같은 라벨로 묶으면 평균·범위를 그림)

산출물: <out>/success.png, yaw_bins.png, margin.png, learning_curve.png, table.md
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .env import ROOT  # noqa: E402


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--tag", default="nominal")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.runs) == len(args.labels)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict]] = defaultdict(list)
    for run, label in zip(args.runs, args.labels):
        d = ROOT / "runs" / run
        ev = d / f"eval_{args.tag}"
        groups[label].append({
            "run": run,
            "summary": json.loads((ev / "summary.json").read_text(encoding="utf-8")),
            "rows": read_csv(ev / "episodes_official.csv") + (read_csv(ev / "episodes_extra.csv")
                                                              if (ev / "episodes_extra.csv").exists() else []),
            "hist": read_csv(d / "eval_history.csv") if (d / "eval_history.csv").exists() else [],
        })
    labels = list(groups)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # 1) 성공률 (공식 10회 / 추가 무작위)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    w = 0.8 / len(labels)
    for i, lb in enumerate(labels):
        off = [g["summary"]["official"]["success_rate"] for g in groups[lb]]
        ext = [g["summary"]["extra"]["success_rate"] if g["summary"]["extra"] else np.nan for g in groups[lb]]
        means = [np.mean(off), np.nanmean(ext)]
        errs = [np.ptp(off) / 2 if len(off) > 1 else 0, np.ptp(ext) / 2 if len(ext) > 1 else 0]
        ax.bar(x + i * w, means, w, yerr=errs, label=lb, color=colors[i % 4], capsize=4)
    ax.set_xticks(x + w * (len(labels) - 1) / 2, ["Seed 0-9 (official)", "extra random"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title(f"Success rate ({args.tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "success.png", dpi=150)

    # 2) yaw 구간별 성공률
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, lb in enumerate(labels):
        bins = groups[lb][0]["summary"]["all"]["yaw_bins"]
        rates = np.array([[b["success_rate"] if b["success_rate"] is not None else np.nan
                           for b in g["summary"]["all"]["yaw_bins"]] for g in groups[lb]])
        ax.plot(range(len(bins)), np.nanmean(rates, 0), "o-", label=lb, color=colors[i % 4])
    ax.set_xticks(range(len(bins)), [b["bin"] for b in bins], rotation=30)
    ax.set_xlabel("initial cube yaw [deg]")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Success rate by initial yaw")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "yaw_bins.png", dpi=150)

    # 3) 성공 회차의 놓은 위치 여유
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [[float(r["final_margin_mm"]) for g in groups[lb] for r in g["rows"] if r["success"] == "True"]
            for lb in labels]
    try:
        ax.boxplot([d if d else [np.nan] for d in data], tick_labels=labels)
    except TypeError:
        ax.boxplot([d if d else [np.nan] for d in data], labels=labels)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("min corner margin to goal edge [mm]")
    ax.set_title("Placement margin (successful trials)")
    fig.tight_layout()
    fig.savefig(out / "margin.png", dpi=150)

    # 4) 학습 곡선 (검증 성공률)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, lb in enumerate(labels):
        for j, g in enumerate(groups[lb]):
            if g["hist"]:
                t = [int(h["timesteps"]) for h in g["hist"]]
                s = [float(h["success_rate"]) for h in g["hist"]]
                ax.plot(t, s, color=colors[i % 4], alpha=0.9, label=lb if j == 0 else None)
    ax.set_xlabel("timesteps")
    ax.set_ylabel("validation success rate")
    ax.set_title("Learning curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=150)

    # 5) 표
    lines = ["| 방법 | 학습 run | 공식 10회 | 추가 무작위 | 한 번에 성공 (전체) | 성공 여유 평균/최소 [mm] | 평균 성공 시간 [s] | 종료 원인 |",
             "|---|---|---|---|---|---|---|---|"]
    for lb in labels:
        for g in groups[lb]:
            s = g["summary"]
            o, e, a = s["official"], s["extra"], s["all"]
            fmt = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
            oneshot = a.get("oneshot_success_rate")
            if oneshot is None and g["rows"]:
                oneshot = float(np.mean([r["success"] == "True" and int(r["drops"]) == 0 for r in g["rows"]]))
            lines.append(f"| {lb} | {g['run']} | {o['successes']}/{o['episodes']} | "
                         f"{(e['success_rate'] * 100 if e else float('nan')):.1f}% | "
                         f"{(oneshot * 100 if oneshot is not None else float('nan')):.1f}% | "
                         f"{fmt(a['success_margin_mm_mean'])} / {fmt(a['success_margin_mm_min'])} | "
                         f"{fmt(a['success_time_s_mean'])} | {a['end_reasons']} |")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
