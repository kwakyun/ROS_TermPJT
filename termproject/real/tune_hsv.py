"""배정 색 큐브의 HSV 범위를 슬라이더로 찾는 도구.

python3 real/tune_hsv.py --image /tmp/topcam.png
 - 슬라이더를 움직여 큐브만 흰색으로 남게 맞춘 뒤 q 를 누르면 real.yaml 에 붙여 넣을 값을 출력한다.
 - 조명이 바뀌면(실습실 낮/밤) 다시 맞춘다. 여러 장으로 확인 권장.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    img = cv2.imread(args.image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    win = "hsv"
    cv2.namedWindow(win)
    for name, val, mx in (("H lo", 40, 180), ("H hi", 85, 180), ("S lo", 80, 255),
                          ("S hi", 255, 255), ("V lo", 60, 255), ("V hi", 255, 255)):
        cv2.createTrackbar(name, win, val, mx, lambda _: None)
    while True:
        g = lambda n: cv2.getTrackbarPos(n, win)  # noqa: E731
        lo = np.array([g("H lo"), g("S lo"), g("V lo")], np.uint8)
        hi = np.array([g("H hi"), g("S hi"), g("V hi")], np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        cv2.imshow(win, np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    print(f"    - {{low: [{lo[0]}, {lo[1]}, {lo[2]}], high: [{hi[0]}, {hi[1]}, {hi[2]}]}}")


if __name__ == "__main__":
    main()
