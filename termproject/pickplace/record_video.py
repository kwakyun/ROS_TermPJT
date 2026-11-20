"""제출용 시뮬레이션 영상 1개 생성 (TERM_PROJECT 1.4 요구사항 반영).

- 최종 정책·설정으로 Seed 0~9 의 10회 전체를 순서대로 녹화 (성공·실패 모두)
- 회차 내부 편집·배속 없음: 제어 주기 20 Hz = 영상 20 fps → 실제 물리 시간과 1:1
- 매 프레임에 회차 번호·Seed·시뮬레이션 경과 시간·현재 단계 표시
- 회차 사이에는 1초 안내 화면(초기화 구간만 줄임), 마지막에 10회 결과·전체 성공률 표
- 왼쪽: top_view 카메라(작업판 수직 탑뷰, 과제 안내 그림과 같은 방향: 위쪽 +X, 왼쪽 +Y), 오른쪽: 비스듬한 자유 시점

예) python -m pickplace.record_video --run A_sym_s7 --out videos/final_sim_10trials.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

from .env import ROOT, PickPlaceEnv
from .evaluate import load_policy, run_episode

W, H = 640, 480
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.6, color=(255, 255, 255), thick=1):
    (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 4), (x + w + 4, y + base + 2), (25, 25, 25), -1)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


class Recorder:
    def __init__(self, env: PickPlaceEnv, path: Path, fps: int) -> None:
        self.env = env
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * W, H))
        self.fps = fps
        self.renderer = mujoco.Renderer(env.model, height=H, width=W)
        self.free = mujoco.MjvCamera()
        self.free.lookat[:] = [0.33, 0.0, 0.02]
        self.free.distance, self.free.azimuth, self.free.elevation = 0.55, 145.0, -32.0
        self.trial = self.seed = 0

    def _render(self, cam) -> np.ndarray:
        self.renderer.update_scene(self.env.data, camera=cam)
        img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        if cam == "top_view":
            # 제공 top_view 는 화면 위쪽이 -X. TERM_PROJECT 탑뷰 그림(위쪽 +X, 왼쪽 +Y)과 맞추려고 180° 회전
            img = cv2.rotate(img, cv2.ROTATE_180)
        return img

    def frame(self, env: PickPlaceEnv, info: dict) -> None:
        img = np.hstack([self._render("top_view"), self._render(self.free)])
        _text(img, f"Trial {self.trial}/10   Seed {self.seed}", (12, 28), 0.75, (0, 255, 255), 2)
        _text(img, f"sim time {info['sim_time']:5.2f} s / 30.00 s", (12, 56))
        _text(img, f"stage: {info['stage_name']}", (12, 82))
        if env.tracker.stage == 4:
            _text(img, f"hold {env.tracker.hold_elapsed:4.2f} / 2.00 s", (12, 108), 0.6, (0, 255, 0))
        sx, sy = info["start_xy"]
        _text(img, f"start ({sx:.3f}, {sy:+.3f}) yaw {np.degrees(info['start_yaw']):+.1f} deg",
              (12, H - 14), 0.5)
        _text(img, "goal: 50x50 mm @ (0.390, 0.000)  |  real-time 1x (20 Hz)", (W + 12, H - 14), 0.5)
        self.writer.write(img)

    def card(self, lines: list[str], seconds: float, colors=None) -> None:
        img = np.full((H, 2 * W, 3), 30, np.uint8)
        step = min(38, (H - 60) // max(1, len(lines)))
        scale = 0.8 if step >= 38 else 0.62
        for i, s in enumerate(lines):
            c = colors[i] if colors else (255, 255, 255)
            _text(img, s, (60, 50 + step * i), scale, c, 2)
        for _ in range(int(seconds * self.fps)):
            self.writer.write(img)

    def close(self) -> None:
        self.writer.release()
        self.renderer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    run_dir = ROOT / "runs" / args.run
    cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    model = load_policy(run_dir / f"policy_{args.which}.zip", cfg["training"]["algorithm"])
    env = PickPlaceEnv(config=cfg)
    seeds = args.seeds if args.seeds else cfg["evaluation"]["official_seeds"]
    out = Path(args.out) if args.out else ROOT / "videos" / f"{args.run}_10trials.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = Recorder(env, out, env.metadata["render_fps"])
    rows = []
    for k, s in enumerate(seeds, start=1):
        rec.trial, rec.seed = k, s
        rec.card([f"Trial {k}/{len(seeds)}   Seed {s}", "Advanced: single-cube pick-and-place",
                  f"policy: {args.run}/policy_{args.which}.zip"], 1.0)
        r = run_episode(model, env, s, frame_cb=rec.frame)
        rows.append(r)
        res = "SUCCESS" if r["success"] else f"FAIL ({r['end_reason']})"
        rec.card([f"Trial {k}  Seed {s}: {res}", f"time {r['sim_time_s']:.2f} s   margin {r['final_margin_mm']:+.1f} mm"],
                 1.0, [(0, 255, 0) if r["success"] else (0, 0, 255), (255, 255, 255)])
        print(f"trial {k} seed {s}: {res}")
    n = sum(r["success"] for r in rows)
    lines = ["RESULT (sim, final policy)"] + [
        f"#{k:>2} seed {r['seed']:>2}  ({r['start_x']:.3f},{r['start_y']:+.3f},{r['start_yaw_deg']:+6.1f})  "
        f"{'SUCCESS' if r['success'] else 'FAIL ' + r['end_reason']}" for k, r in enumerate(rows, 1)
    ] + [f"success rate {n}/{len(rows)} = {100 * n / len(rows):.0f} %"]
    colors = [(0, 255, 255)] + [(0, 255, 0) if r["success"] else (0, 0, 255) for r in rows] + [(0, 255, 255)]
    rec.card(lines, 6.0, colors)
    rec.close()
    print(f"saved {out}  ({n}/{len(rows)})")


if __name__ == "__main__":
    main()
