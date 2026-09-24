"""Utility helpers: model download and frame annotation."""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from touchguard.config import UIConfig
from touchguard.datatypes import (
    FINGERTIP_INDICES,
    FaceBox,
    HandLandmarks,
    TouchStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe left-hand topology (wrist + 5 fingers x 4 joints).
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# BGR colors per TouchGuard status.
STATUS_COLORS: dict[TouchStatus, tuple[int, int, int]] = {
    TouchStatus.SAFE: (70, 190, 70),
    TouchStatus.HAND_APPROACHING: (0, 165, 255),
    TouchStatus.FACE_TOUCH_DETECTED: (40, 60, 240),
    TouchStatus.NO_FACE_DETECTED: (90, 90, 200),
}

LANDMARK_COLOR = (255, 160, 0)  # bright cyan skeleton, visible over skin
LANDMARK_TIP_COLOR = (0, 60, 255)  # red fingertips
REGION_COLOR = (255, 255, 255)  # white region anchors
DISTANCE_LINE_COLOR = (0, 255, 255)  # yellow fingertip -> face region


def status_label(status: TouchStatus) -> str:
    """ASCII-safe status text for consoles and OpenCV rendering.

    The glossy unicode glyphs in :class:`TouchStatus` (e.g. ``⚠``) cannot be
    rendered by OpenCV's Hershey fonts and crash some Windows consoles, so
    they are replaced here.
    """
    return status.value.replace("⚠", "!! ").replace("–", "-")


def status_token(status: TouchStatus) -> str:
    """Short, greppable status token for the per-second console line.

    Maps each :class:`TouchStatus` to a single word so diagnostic output is
    easy to parse at a glance:

    * ``SAFE``       - hand away from the face, system armed
    * ``APPROACHING``- hand between the approach and touch thresholds
    * ``TOUCHING``   - confirmed face-touch event (count was incremented)
    * ``NO FACE``    - no face in frame, no touch possible
    """
    return {
        TouchStatus.SAFE: "SAFE",
        TouchStatus.HAND_APPROACHING: "APPROACHING",
        TouchStatus.FACE_TOUCH_DETECTED: "TOUCHING",
        TouchStatus.NO_FACE_DETECTED: "NO FACE",
    }.get(status, status_label(status))


def ensure_model_exists(model_path: str | Path, url: str, timeout: float = 30.0) -> bool:
    """Ensure a model file exists locally, downloading it when necessary.

    Returns ``True`` when the file is present and non-empty. Never raises:
    download failures are logged and result in ``False`` so callers can
    degrade gracefully.
    """
    path = Path(model_path)
    if path.exists() and path.stat().st_size > 0:
        return True
    if not url:
        logger.warning("No download URL provided for model '%s'.", path.name)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".download")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()
        if not data:
            raise ValueError("downloaded model is empty")
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
        logger.info("Downloaded model '%s'.", path.name)
        return True
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("Could not download model '%s' from %s: %s", path.name, url, exc)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        return False


def annotate_frame(
    frame: Any,
    faces: list[FaceBox],
    hands: list[HandLandmarks],
    status: TouchStatus,
    ui_config: UIConfig | None = None,
    fps: float | None = None,
    touch_count: int = 0,
    distance: float | None = None,
    closest_point: tuple[float, float] | None = None,
    region_point: tuple[float, float] | None = None,
    region_name: str | None = None,
) -> Any:
    """Draw detection overlays and status text onto a copy of the frame.

    If OpenCV is not importable the original frame is returned untouched.
    """
    try:
        import cv2
    except ImportError:
        return frame

    annotated = frame.copy()
    ui_config = ui_config or UIConfig()
    color = STATUS_COLORS.get(status, STATUS_COLORS[TouchStatus.SAFE])

    for face in faces:
        x1, y1, x2, y2 = face.to_int_tuple()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, "FACE", (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
        if ui_config.show_regions:
            for region in face.regions():
                rx, ry = int(region.point[0]), int(region.point[1])
                cv2.circle(annotated, (rx, ry), 4, REGION_COLOR, -1, cv2.LINE_AA)
                cv2.putText(
                    annotated, region.label, (rx + 6, ry - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, REGION_COLOR, 1, cv2.LINE_AA,
                )

    if ui_config.show_landmarks:
        for hand in hands:
            pts = hand.points
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    p1 = (int(pts[a][0]), int(pts[a][1]))
                    p2 = (int(pts[b][0]), int(pts[b][1]))
                    cv2.line(annotated, p1, p2, LANDMARK_COLOR, 1, cv2.LINE_AA)
            for index, point in enumerate(pts):
                cx, cy = int(point[0]), int(point[1])
                is_tip = index in FINGERTIP_INDICES
                dot_color = LANDMARK_TIP_COLOR if is_tip else (255, 255, 255)
                cv2.circle(
                    annotated, (cx, cy), 8 if is_tip else 3, dot_color, -1, cv2.LINE_AA
                )
            if hand.label:
                cv2.putText(
                    annotated, hand.label, (int(pts[0][0]) - 20, int(pts[0][1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA,
                )

    # Distance indicator line from the closest landmark to the nearest region.
    if closest_point is not None and distance is not None:
        tip = (int(closest_point[0]), int(closest_point[1]))
        if region_point is not None:
            reg = (int(region_point[0]), int(region_point[1]))
        else:
            reg = tip
        cv2.line(annotated, tip, reg, DISTANCE_LINE_COLOR, 2, cv2.LINE_AA)
        label = region_name.replace("_", " ").upper() if region_name else "FACE"
        cv2.putText(
            annotated, label, (tip[0] + 8, tip[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, DISTANCE_LINE_COLOR, 1, cv2.LINE_AA,
        )

    _draw_info_panel(annotated, status, ui_config, touch_count, distance, fps)

    if status == TouchStatus.FACE_TOUCH_DETECTED:
        _draw_warning_banner(annotated)

    return annotated


def _draw_info_panel(
    frame: Any,
    status: TouchStatus,
    ui_config: UIConfig,
    touch_count: int,
    distance: float | None,
    fps: float | None,
) -> None:
    import cv2

    color = STATUS_COLORS.get(status, STATUS_COLORS[TouchStatus.SAFE])
    distance_text = (
        f"{distance:.0f} px" if distance is not None and distance != float("inf")
        else "N/A"
    )
    rows = [
        ("Status", status_label(status), color),
        ("Face Touch Count", str(touch_count), (240, 240, 240)),
        ("Hand -> Face", distance_text, (240, 240, 240)),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_x, panel_y, line_h, pad = 8, 8, 22, 10
    panel_w = max(
        cv2.getTextSize(f"{label}: {value}", font, 0.55, 1)[0][0]
        for label, value, _ in rows
    ) + pad
    panel_h = line_h * len(rows) + pad
    cv2.rectangle(
        frame, (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1,
    )
    cv2.rectangle(
        frame, (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h), color, 1,
    )
    for i, (label, value, value_color) in enumerate(rows):
        y = panel_y + pad + i * line_h + 14
        cv2.putText(
            frame, label + ":", (panel_x + pad, y),
            font, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )
        label_w = cv2.getTextSize(label + ":", font, 0.55, 1)[0][0]
        cv2.putText(
            frame, value, (panel_x + pad + label_w + 8, y),
            font, 0.55, value_color, 1, cv2.LINE_AA,
        )

    if fps is not None and ui_config.show_fps:
        cv2.putText(
            frame, f"FPS: {fps:.0f}", (frame.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA,
        )


def _draw_warning_banner(frame: Any) -> None:
    import cv2

    height, width = frame.shape[:2]
    bar_h = 50
    y0 = max(height - bar_h, 0)
    cv2.rectangle(frame, (0, y0), (width, height), (40, 60, 240), -1)
    cv2.rectangle(frame, (0, y0), (width, height), (0, 0, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "!! FACE TOUCH ALERT - WARNING !!"
    (tw, th), _ = cv2.getTextSize(text, font, 0.9, 2)
    cv2.putText(
        frame, text, ((width - tw) // 2, y0 + 24),
        font, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
    )
    sub = "PLEASE KEEP YOUR HANDS AWAY FROM YOUR FACE"
    (sw, _), _ = cv2.getTextSize(sub, font, 0.45, 1)
    cv2.putText(
        frame, sub, ((width - sw) // 2, y0 + 42),
        font, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )