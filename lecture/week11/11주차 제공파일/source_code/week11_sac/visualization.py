"""Topcam overlays for object, target, moving fingertip, and EEF."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .env import MovingJawContactEnv


OBJECT_COLOR = (40, 225, 40)
TARGET_COLOR = (255, 120, 30)
MOVING_TIP_COLOR = (255, 0, 255)
EEF_COLOR = (0, 255, 255)
SCENARIO_COLORS = {
    "straight": (170, 170, 255),
    "diagonal_left": (255, 160, 80),
    "diagonal_right": (160, 100, 255),
}


def _point(uv: np.ndarray) -> tuple[int, int]:
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def _outlined_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.45,
) -> None:
    cv2.putText(
        frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
        scale, (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA,
    )


def _draw_spawn_region(
    frame: np.ndarray,
    polygon: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> None:
    if polygon.shape != (4, 2) or not np.all(np.isfinite(polygon)):
        return
    if np.max(np.ptp(polygon, axis=0)) < 1.0:
        return
    points = np.rint(polygon).astype(np.int32).reshape((-1, 1, 2))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0.0, frame)
    cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
    label_point = tuple(points[0, 0, :].tolist())
    _outlined_text(
        frame, label,
        (max(4, label_point[0] + 3), max(16, label_point[1] - 5)),
        color, 0.34,
    )


def _draw_cube(
    frame: np.ndarray, environment: MovingJawContactEnv, *, target: bool
) -> None:
    color = TARGET_COLOR if target else OBJECT_COLOR
    center = environment.cube_pixel(target)
    lower, upper = environment.cube_bbox_pixels(target)
    if not all(np.all(np.isfinite(value)) for value in (center, lower, upper)):
        return
    cv2.rectangle(frame, _point(lower), _point(upper), color, 2, cv2.LINE_AA)
    cv2.circle(frame, _point(center), 4, color, -1, cv2.LINE_AA)
    _outlined_text(
        frame, "TARGET" if target else "OBJECT",
        (_point(upper)[0] + 4, _point(lower)[1] + 12), color, 0.35,
    )


def _draw_landmark(
    frame: np.ndarray,
    uv: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    if uv.shape != (2,) or not np.all(np.isfinite(uv)):
        return
    height, width = frame.shape[:2]
    if not (0 <= uv[0] < width and 0 <= uv[1] < height):
        return
    center = _point(uv)
    cv2.circle(frame, center, radius, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)
    cv2.line(
        frame, (center[0] - radius - 3, center[1]),
        (center[0] + radius + 3, center[1]), color, 2, cv2.LINE_AA,
    )
    cv2.line(
        frame, (center[0], center[1] - radius - 3),
        (center[0], center[1] + radius + 3), color, 2, cv2.LINE_AA,
    )
    _outlined_text(frame, label, (center[0] + 13, center[1] - 10), color)


def top_view_bgr(
    environment: MovingJawContactEnv,
    lines: Iterable[str] = (),
) -> np.ndarray:
    rgb = environment.render_camera("top_view")
    if rgb is None:
        raise RuntimeError("environment requires render_mode='rgb_array'")
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    object_regions = (
        ((None, "OBJ ALL", OBJECT_COLOR),)
        if environment.uses_shared_object_region
        else tuple(
            (
                scenario,
                f"OBJ {scenario}",
                SCENARIO_COLORS.get(scenario, OBJECT_COLOR),
            )
            for scenario in environment.scenario_names
        )
    )
    for scenario, label, color in object_regions:
        _draw_spawn_region(
            frame,
            environment.spawn_region_pixels(target=False, scenario=scenario),
            label,
            color,
        )
    _draw_spawn_region(
        frame,
        environment.spawn_region_pixels(target=True),
        "TARGET",
        TARGET_COLOR,
    )
    _draw_cube(frame, environment, target=False)
    _draw_cube(frame, environment, target=True)
    _draw_landmark(
        frame, environment.eef_pixel(), "EEF/IK", EEF_COLOR, radius=10
    )
    _draw_landmark(
        frame, environment.moving_tip_pixel(), "MOVING TIP", MOVING_TIP_COLOR,
        radius=12,
    )
    for index, line in enumerate(lines):
        _outlined_text(
            frame, str(line), (8, 22 + index * 23), (245, 245, 245), 0.48
        )
    return frame


def front_view_bgr(environment):
    rgb = environment.render_camera("front_view")
    if rgb is None:
        raise RuntimeError("environment requires render_mode=rgb_array")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
