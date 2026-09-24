# TouchGuard AI — Design Document

## System Architecture

TouchGuard AI is built around a modular, UI-agnostic detection core. Each
responsibility lives in its own module so components can be unit-tested in
isolation and swapped without touching the rest of the system.

```
                          +---------------------------+
   webcam                 |  VideoStream              |
   frames --------------> |  (graceful open/read/     |
                          |   release, CameraError)   |
                          +-------------+-------------+
                                        |
                                        v
                 +----------------------------------------+
                 |  TouchGuardPipeline (orchestrator)     |
                 |  process(frame) -> FrameAnalysis        |
                 +----------------------------------------+
                    |          |              |           |
                    v          v              v           v
            +------------+ +-----------+ +----------+ +---------------+
            | FaceDetector| | HandDet.  | | Touch    | | VoiceWarning  |
            | (Haar case) | | (MediaPipe| | Logic    | | Service (TTS) |
            |             | | Tasks API)| | distance | |  threaded     |
            +------------+ +-----------+ +----------+ +---------------+
                    \          |              |           /
                     \         v              v          /
                      +--------+ FrameAnalysis + annotated frame
                      +--------> (status, distance, touch_count, warn)
                                |
                                v
                      +------------------+
                      | OpenCV GUI app.py |
                      +------------------+
```

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `touchguard/config.py` | Typed settings, YAML loading, environment overrides |
| `touchguard/datatypes.py` | Shared DTOs (`FaceBox`, `FaceRegion`, `HandLandmarks`, `TouchStatus`) |
| `touchguard/face_detector.py` | Haar-cascade face detection with model auto-download |
| `touchguard/hand_detector.py` | MediaPipe Hand Landmarker (Tasks API) landmark tracking |
| `touchguard/face_touch_detector.py` | Pure, deterministic distance/event state machine |
| `touchguard/voice_warning.py` | Debounced, threaded text-to-speech warnings |
| `touchguard/video_stream.py` | Webcam abstraction with graceful failure handling |
| `touchguard/pipeline.py` | Orchestrates one frame of processing |
| `touchguard/utils.py` | Frame annotation, model download helpers |

## Distance-based event detection

The touch logic computes, for every fingertip (and optionally the wrist), the
**minimum pixel distance to the face**: the distance to the nearest point on
the face rectangle (0 inside) plus the distance to the five face-region anchors
(forehead, nose, left/right cheek, mouth/chin).

Zones (configurable, default values relative to a 640×480 frame):

```text
distance > APPROACH (170)            -> SAFE
APPROACH >= distance > TOUCH (90)    -> HAND APPROACHING
distance <= TOUCH (90)               -> FACE TOUCH DETECTED
```

Event rules (one warning per NEW touch, no repeats while the hand stays):

1. A candidate touch must persist `persist_frames` consecutive frames
   (debounce against single-frame false positives).
2. The first confirmed touch increments the touch counter and fires the voice
   warning exactly once. Continuous contact is ONE touch event, so a long hold
   never floods warnings (`voice.repeat_interval_seconds` defaults to 0; set
   it above 0 to re-announce periodically while the hand stays).
3. The system only re-arms once the hand has moved beyond
   `touch + release_margin` pixels (default 90 + 30 = 120 px), for
   `clean_frames` consecutive frames (hysteresis). This guarantees exactly one
   warning per continuous touch and a fresh warning for every new touch after
   the hand is removed.
4. As a second layer, the voice service enforces a minimum interval between
   spoken warnings (default 0 s, so a new touch is never silently dropped).

## Real-time loop

```
while monitoring:
    frame = stream.read()
    analysis = pipeline.process(frame)   # detect -> decide -> warn -> annotate
    display(analysis.frame)              # cv2.imshow
    handle keyboard / UI controls        # q quit, p pause, m mute, r reset
```

FPS is reported from a smoothed frame-time EMA that restarts on large
speed-ups, so the expensive model-initialisation frame never drags the
displayed FPS down.

## Model management

| Model | Path | Auto-download URL |
| --- | --- | --- |
| Face Haar cascade | `assets/models/haarcascade_frontalface_default.xml` | OpenCV GitHub |
| Hand Landmarker | `models/hand_landmarker.task` (float16) | Google Storage (MediaPipe models) |

Both fall back to an **ASCII-only temp path** at runtime: OpenCV's and
MediaPipe's C++ file APIs cannot open files whose absolute path contains
non-ASCII characters (e.g. an en-dash in the folder name), so the model is
copied to `%TEMP%/touchguard/` and loaded from there.

## Graceful degradation rules

- Webcam cannot be opened: `CameraError` is raised, `run_app` prints a clear
  message and returns a non-zero exit code.
- Haar model missing / download fails: face detection is disabled for the
  session, other subsystems continue.
- MediaPipe / Hand Landmarker model missing or unreadable: hand detection is
  disabled, remaining subsystems run and the reason is reported.
- TTS engine unavailable: the voice service falls back to a short speaker
  beep.
- No face / no hand at any frame: the state machine returns `NO FACE DETECTED`
  or `SAFE` without warnings — it never crashes when a hand temporarily leaves
  the frame.