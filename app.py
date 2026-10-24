"""TouchGuard AI - desktop (OpenCV GUI) application entry point.

Run with::

    python app.py

The webcam opens in a live OpenCV window that shows the face box, face
regions, hand skeleton and fingertip, the current status, the face-to-hand
distance and the total touch count. Any confirmed face touch triggers an
immediate voice warning: *"Warning! Please keep your hands away from your
face."*

Controls:

* ``q``  - quit
* ``p``  - pause / resume monitoring
* ``m``  - mute / unmute voice warnings
* ``r``  - reset detection event state (touch counter included)
* ``t``  - test: force one voice warning now (verifies the audio works)

``python app.py --diagnose`` prints live face/hand detection counts and the
voice engine status so you can quickly tell which subsystem is not detecting.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

from touchguard.config import Settings, load_settings
from touchguard.exceptions import TouchGuardError
from touchguard.pipeline import TouchGuardPipeline
from touchguard.utils import status_token
from touchguard.video_stream import VideoStream

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python app.py",
        description="TouchGuard AI - Real-Time Face-Touch Detection & Voice Warning System",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML configuration file (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Webcam index to use (overrides the configuration file).",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Live diagnostic mode: print per-frame face/hand detection "
        "counts and voice-engine status to find why alerts do not fire.",
    )
    return parser


def cv2_gui_available() -> bool:
    """Return True when the installed OpenCV build supports HighGUI windows."""
    try:
        import cv2
    except ImportError:
        return False
    try:
        info = cv2.getBuildInformation() or ""
        for line in info.splitlines():
            if line.strip().startswith("GUI:"):
                return "NONE" not in line.upper()
    except Exception:
        pass
    return True


def _print_model_status(pipeline: TouchGuardPipeline) -> None:
    for name, info in pipeline.model_status.items():
        if name == "voice":
            print(f"[TouchGuard AI] Voice: {info.get('mode', 'n/a')}")
            continue
        if not info.get("available", True):
            print(
                f"[TouchGuard AI] WARNING: {name} detection unavailable -> "
                f"{info.get('reason')}",
                file=sys.stderr,
            )


def _print_diagnose_status(pipeline: TouchGuardPipeline) -> None:
    """Print explicit face/hand/voice availability so gaps are easy to spot."""
    status = pipeline.model_status
    face = status.get("face", {})
    hand = status.get("hand", {})
    voice = status.get("voice", {})
    print(
        "[TouchGuard AI] DIAGNOSE: face="
        f"{'OK' if face.get('available') else 'UNAVAILABLE'} "
        f"hand={'OK' if hand.get('available') else 'UNAVAILABLE'} "
        f"voice={voice.get('mode', 'n/a')}"
    )
    if not face.get("available", True):
        print(f"[TouchGuard AI] DIAGNOSE: face reason -> {face.get('reason')}")
    if not hand.get("available", True):
        print(f"[TouchGuard AI] DIAGNOSE: hand reason -> {hand.get('reason')}")
    print(
        "[TouchGuard AI] DIAGNOSE: Watch the 'Faces:' and 'Hands:' counts "
        "below. The voice alert only fires when a FACE is detected and a HAND "
        "is within the touch zone."
    )


def run_app(
    settings: Settings,
    stream: VideoStream | None = None,
    diagnose: bool = False,
) -> int:
    """Run the desktop monitoring loop; returns a process exit code."""
    try:
        import cv2
    except ImportError:
        print(
            "[TouchGuard AI] ERROR: OpenCV is required. Install dependencies "
            "with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    if not cv2_gui_available():
        print(
            "[TouchGuard AI] ERROR: the installed OpenCV build has no GUI "
            "support (opencv-python-headless) so the video window cannot be "
            "displayed. Install the full GUI build and try again:\n"
            "    pip uninstall opencv-python-headless\n"
            "    pip install opencv-contrib-python",
            file=sys.stderr,
        )
        return 2

    if stream is None:
        stream = VideoStream(
            settings.camera.index,
            settings.camera.width,
            settings.camera.height,
            settings.camera.fps,
        )
    try:
        stream.open()
    except TouchGuardError as exc:
        print(f"[TouchGuard AI] ERROR: {exc}", file=sys.stderr)
        return 1

    pipeline = TouchGuardPipeline(settings)
    paused = False
    analysis = None
    last_log = 0.0

    print("[TouchGuard AI] Monitoring started. Press 'q' to quit.")
    print("[TouchGuard AI] Controls: q=quit p=pause m=mute r=reset t=test voice")
    _print_model_status(pipeline)
    if diagnose:
        _print_diagnose_status(pipeline)
    try:
        while True:
            try:
                frame = stream.read()
            except TouchGuardError as exc:
                print(f"[TouchGuard AI] Camera stream ended: {exc}", file=sys.stderr)
                break

            if not paused:
                analysis = pipeline.process(frame)
                frame = analysis.frame

            cv2.imshow(settings.ui.window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
                print(f"[TouchGuard AI] {'Paused.' if paused else 'Resumed.'}")
            elif key == ord("m"):
                new_state = not settings.voice.enabled
                settings.voice.enabled = new_state
                pipeline.voice.set_enabled(new_state)
                print(f"[TouchGuard AI] Voice warnings {'muted.' if not new_state else 'on.'}")
            elif key == ord("r"):
                pipeline.reset()
                print("[TouchGuard AI] Detection state reset.")
            elif key == ord("t"):
                message = settings.voice.main_warning
                try:
                    message = message.format(
                        count=pipeline.logic.touch_count or 0
                    )
                except (KeyError, IndexError, ValueError):
                    pass
                accepted = pipeline.voice.announce(message)
                print(
                    "[TouchGuard AI] TEST voice warning -> "
                    + ("sent." if accepted else "muted (press 'm' to unmute).")
                )

            now = time.perf_counter()
            if analysis is not None and not paused and now - last_log >= 1.0:
                last_log = now
                distance = (
                    f"{analysis.distance:.0f} px"
                    if analysis.distance != float("inf")
                    else "N/A"
                )
                print(
                    f"[TouchGuard AI] STATUS: {status_token(analysis.status)} | "
                    f"Faces: {len(analysis.faces)} | Hands: {len(analysis.hands)} | "
                    f"FACE TOUCH COUNT: {analysis.touch_count} | "
                    f"Hand -> Face Distance: {distance}"
                )
                if diagnose and (
                    not analysis.faces or not analysis.hands
                ):
                    missing = "face" if not analysis.faces else "hand"
                    print(
                        f"[TouchGuard AI] DIAGNOSE: no {missing} detected this "
                        f"second -> the touch alert cannot fire."
                    )
    finally:
        stream.release()
        pipeline.close()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    _make_console_unicode_safe()
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
    except TouchGuardError as exc:
        print(f"[TouchGuard AI] ERROR: {exc}", file=sys.stderr)
        return 1
    if args.camera is not None:
        settings.camera.index = int(args.camera)
    return run_app(settings, diagnose=args.diagnose)


def _make_console_unicode_safe() -> None:
    """Prevent non-representable glyphs (e.g. ``⚠``) from crashing prints."""
    import codecs

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace", line_buffering=True)
            except Exception:  # pragma: no cover - best-effort
                pass
        else:  # pragma: no cover - very old interpreters
            sys.stdout = codecs.getwriter("utf-8")(stream)  # type: ignore[assignment]


if __name__ == "__main__":
    sys.exit(main())