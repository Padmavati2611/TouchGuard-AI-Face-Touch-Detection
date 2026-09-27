# TouchGuard AI — Real-Time Face-Touch Detection & Voice Warning System

Real-time computer vision that watches your webcam, detects when your hand
touches or approaches your face, and immediately speaks an audible warning —
helping you break the unconscious habit of touching your face.

```text
---------------------------------------------
           TOUCHGUARD AI
---------------------------------------------

Status: ⚠ FACE TOUCH DETECTED

FACE TOUCH COUNT: 4

Hand -> Face Distance: 35 px

[Webcam Feed]
  Face Box  +  Face Regions
  Hand Landmarks  +  Fingertip Marker
---------------------------------------------
```

## 1. Project Overview

TouchGuard AI combines **OpenCV face detection**, **MediaPipe Hand Landmarker**
(tasks API) hand tracking, a dedicated **distance-based face-touch state
machine**, and an offline **text-to-speech warning service**. It monitors a
live webcam feed and alerts the user the instant a hand enters the face zone.

The application is intentionally **face detection only** — it never identifies,
recognises, or stores who you are. All processing happens locally on your
computer. Nothing is uploaded.

## 2. Problem Statement

People touch their faces dozens of times per hour, often unconsciously. Hands
pick up pathogens from surfaces and transfer them to the eyes, nose, and mouth
— a major vector for infection. Because the behaviour is automatic, people are
rarely aware they are doing it, so "reminding yourself" does not work. A
system that **detects the behaviour in real time and responds with an immediate
audible cue** can help retrain the habit.

## 3. Objectives

- Detect face-touching behaviour in real time from a standard webcam.
- Combine face detection, hand-landmark tracking, event detection, and voice
  output in one clean, modular pipeline.
- Emit **one warning per new touch event** — never a flood of repeated
  warnings while the hand stays on the face.
- Minimise false positives through configurable distance thresholds, debounce,
  and release hysteresis.
- Fail gracefully (no crashes) on webcam, model, and dependency failures.
- Stay fully local and privacy-preserving (no face recognition, no uploads).

## 4. Features

| Feature | Detail |
| --- | --- |
| Real-time webcam monitoring | Live, annotated OpenCV preview |
| Face detection | OpenCV Haar cascade, auto-downloaded model |
| Face regions | Forehead, nose, left/right cheek, mouth/chin anchors drawn on the face |
| Hand landmark tracking | MediaPipe Hand Landmarker (Tasks API), 21 points per hand |
| Fingertip tracking | Thumb, index, middle, ring, little fingertips highlighted |
| Distance measurement | Minimum fingertip → face distance in pixels, live |
| Three-zone logic | SAFE → HAND APPROACHING → FACE TOUCH DETECTED |
| Repeated-alert handling | One voice warning per NEW touch; re-arms with hysteresis for the next touch; a continuous hold never floods warnings |
| Voice warning | Offline TTS on a background thread: *"Warning. You touched your face. Touch number 1, 2, 3…"* each new touch |
| Touch counter | Total confirmed face-touch events tracked and displayed |
| Configurable thresholds | `config/settings.yaml` |
| Graceful degradation | Webcam / model / voice failures handled, no crashes |

## 5. Technologies Used

| Layer | Technology |
| --- | --- |
| Language | Python 3.10 (also runs on 3.9–3.12) |
| Computer vision | OpenCV (Haar cascade, video capture, HighGUI window) |
| Hand / landmark tracking | MediaPipe Hand Landmarker (Tasks API) |
| Event detection | Custom distance-based state machine (`touchguard.face_touch_detector`) |
| Voice warning | pyttsx3 (offline text-to-speech) + beep fallback |
| Configuration | YAML (`config/settings.yaml`) |
| Testing | pytest (+ pytest-mock), webcam-free and mocked |
| Privacy | 100% local processing, no cloud, no face recognition |

## 6. How the Detection Works (Workflow)

```text
                +------------------------------+
                |  1. Capture frame from webcam |
                +--------------+---------------+
                               |
                               v
                +--------------+---------------+
                |  2. Detect face (Haar box)   |
                +--------------+---------------+
                               |
                               v
                +--------------+---------------+
                |  3. Detect hand landmarks    |
                |     (21 points, fingertips)  |
                +--------------+---------------+
                               |
                               v
                +--------------+---------------+
                |  4. Distance-based logic     |
                |     nearest fingertip ->     |
                |     face distance (px)       |
                +--------------+---------------+
              safe?     |  approaching?  |  touch?
        +---------------+   +------------+   +-------------------+
        |  SAFE         |   | APPROACHING|   | FACE TOUCH + voice |
        +---------------+   +------------+   +-------------------+
                               ^                                  ^
                               |             remove hand          |
                               +-------------------------------------> re-arm for next touch
```

## 7. Face Detection Explanation

Face detection is performed with OpenCV's **Haar cascade classifier**
(`haarcascade_frontalface_default.xml`). It is a classical, fast, CPU-friendly
detector: a cascade of weak classifiers trained to find frontal faces in a
grayscale image.

- The frame is converted to grayscale and histogram-equalised.
- `detectMultiScale` slides a window over the image at multiple scales,
  applying the cascade. `scale_factor` (1.1) and `min_neighbors` (5) trade
  sensitivity vs. false positives.
- On high-resolution frames, the image is downscaled to at most
  `max_detect_width` (640 px) before detection and the boxes are scaled back,
  keeping the loop real-time.
- Detection is continuous, so the user may move left/right or closer/farther.

The Haar model is downloaded automatically on first run (into
`assets/models/`) when missing.

## 8. Hand Detection Explanation

Hand tracking uses MediaPipe's supported **Hand Landmarker** (Tasks API) with
the `models/hand_landmarker.task` model in VIDEO running mode.

Each detected hand returns **21 normalised landmarks**: wrist (0), thumb (1–4),
index (5–8), middle (9–12), ring (13–16), and little (17–20) finger. The
landmarks are multiplied by the frame size to obtain pixel coordinates, then
the five fingertips (indices **4, 8, 12, 16, 20**) are used for face-touch
detection, with the **index fingertip (8)** being the primary one.

The model file is auto-downloaded on first use into `models/` when missing.

## 9. Face-Touch Detection Logic

For every frame, and for every fingertip (plus the wrist if enabled), the
system computes the **minimum pixel distance from that point to the face** —
defined as the distance to the nearest point on the face rectangle (0 when the
point is inside the box) and to the forehead/nose/cheek/mouth-chin region
anchors.

The three zones are defined by two configurable thresholds (in `settings.yaml`):

```python
APPROACH_DISTANCE_THRESHOLD = 170   # px  -> closer than this: hand approaching
TOUCH_DISTANCE_THRESHOLD = 90       # px  -> closer than this: face touch
VOICE_COOLDOWN = 3                  # s   -> min seconds between reminders
```

```text
distance > APPROACH  -> SAFE
APPROACH >= distance > TOUCH  -> HAND APPROACHING FACE
distance <= TOUCH    -> ⚠ FACE TOUCH DETECTED  (+ voice warning)
```

**Debounce:** a touch must persist for `persist_frames` consecutive frames
before it is confirmed (default 3, ~0.1 s), which filters single-frame jitter
from normal hand movement near the face.

## 10. Voice Warning Mechanism

On a confirmed touch the pipeline schedules a warning on a **background
worker thread**, so the camera loop never blocks on speech synthesis:

```text
Warning. You touched your face. Touch number 1.
```

The **same message is spoken once for every new face-touch event**, with the
current touch number filled in — a continuous touch is one event and never
floods warnings. The running face-touch total is shown live on the overlay and
in the console, so you can see "FACE TOUCH COUNT: 1", "2", "3", ... as each new
touch occurs.
Text-to-speech runs offline via **pyttsx3** (Windows SAPI, macOS NSSpeech,
  Linux eSpeak).
- If the TTS engine cannot be initialised, the service falls back to a short
  speaker beep (configurable) instead of crashing.
- The cooldown (`min_interval_seconds`) defaults to 0 so a quick
  touch → remove → touch cycle is **never silently dropped**; every separate
  touch gets its own warning.

## 11. Repeated-Alert Mechanism (one warning per NEW touch)

The system is **event-based**, not frame-based. A state machine with
**hysteresis** ensures each *new* touch generates a *new* warning while one
continuous touch never triggers a flood.

```python
if distance <= TOUCH_DISTANCE_THRESHOLD:
    if not touch_active:
        touch_active = True          # first contact -> ONE warning
        touch_count += 1
        speak_warning()
    # continuous contact -> touch_active stays True -> no repeated warning
elif distance > TOUCH_DISTANCE_THRESHOLD + RELEASE_MARGIN:
    touch_active = False             # hand removed -> system re-arms
```

```text
 SAFE  ->  APPROACHING  ->  FACE TOUCH DETECTED  ->  WARNING (each new touch)
   ^                                                     |
   +------------------ hand removed ---------------------+
      -> SAFE -> NEW TOUCH -> WARNING AGAIN (with updated count)
```

The **same warning text** ("Warning. You touched your face. Touch number {count}.")
is spoken once for every new touch, numbered with the running total
(`FACE TOUCH COUNT: 1`, `2`, `3`, ...) which is displayed live on the overlay and
in the console so you always know how many separate touches have been detected.
A continuous touch is ONE event - it is never re-announced while the hand
stays on the face (set `repeat_interval_seconds` > 0 only if you want periodic
reminders).

Two extra layers prevent false re-arming:

- **Release margin** (`release_margin`, default 30 px): the hand must move
  back further than `touch_distance_threshold + release_margin` (90 + 30 =
  120 px) before the system re-arms, so a hand hovering at the boundary cannot
  re-trigger.
- **Clean frames** (`clean_frames`, default 3): the release must hold for
  several consecutive frames before re-arming.

The touch counter (`FACE TOUCH COUNT`) increments exactly once per confirmed
event, and the counter appears live on the overlay and in the console.

## 12. Project Structure

```text
TouchGuard_AI/
│
├── app.py                        # Desktop (OpenCV GUI) entry point: python app.py
├── main.py                       # Backward-compatible alias for python app.py
├── requirements.txt              # Runtime dependencies (Python 3.10)
├── requirements-dev.txt          # Test dependencies
├── pyproject.toml                # Packaging metadata
├── pytest.ini
├── .gitignore
│
├── config/
│   └── settings.yaml             # All tunable thresholds & options
│
├── models/                       # MediaPipe hand landmarker model
│   ├── hand_landmarker.task      # (auto-downloaded on first run)
│   └── README.md
│
├── assets/
│   └── models/                   # Haar cascade face model (auto-managed)
│
├── scripts/
│   └── download_models.py        # Pre-download face + hand models
│
├── touchguard/                   # Core, UI-agnostic package
│   ├── __init__.py
│   ├── config.py                 # Typed settings + YAML loader
│   ├── datatypes.py              # FaceBox, FaceRegion, HandLandmarks, TouchStatus
│   ├── exceptions.py             # Custom error hierarchy
│   ├── face_detector.py          # Haar cascade face detection
│   ├── hand_detector.py          # MediaPipe Hand Landmarker (Tasks API)
│   ├── face_touch_detector.py    # Distance-based touch/event state machine
│   ├── voice_warning.py          # Threaded TTS with cooldown/fallback
│   ├── video_stream.py           # Webcam abstraction
│   ├── pipeline.py               # Orchestrator (one frame -> FrameAnalysis)
│   └── utils.py                  # Annotation + model download helpers
│
├── docs/
│   └── DESIGN.md                 # Architecture document
│
└── tests/                        # Webcam-free pytest suite (mocked)
    ├── conftest.py
    ├── test_config.py
    ├── test_face_touch_detector.py
    ├── test_face_detector.py
    ├── test_hand_detector.py
    ├── test_voice_warning.py
    ├── test_video_stream.py
    ├── test_pipeline.py
    ├── test_utils.py
    └── test_app_startup.py
```

## 13. Installation Instructions

**Prerequisites**

- Python 3.10 (recommended; 3.9–3.12 should also work)
- A working webcam

```bash
# 1) Clone / open the project folder
cd "TouchGuard AI – Real-Time Face-Touch Detection & Voice Warning System"

# 2) Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) (Recommended) Pre-download the models
python scripts/download_models.py
```

> The Haar cascade and the hand landmarker `.task` file are also downloaded
> automatically on first run if you skip step 4 (requires internet).

## 14. How to Run

```bash
# Windows PowerShell: activate the venv first, then...
python app.py
```

The webcam window opens with all overlays. Controls:

| Key | Action |
| --- | --- |
| `q` | Quit |
| `p` | Pause / resume monitoring |
| `m` | Mute / unmute voice warnings |
| `r` | Reset detection state (touch counter included) |
| `t` | Test voice warning now (verifies the audio works) |

Command line options:

```bash
python app.py --camera 1        # use a different webcam index
python app.py --config my.yaml  # use a custom configuration file
python app.py --diagnose        # live diagnostics: face/hand detection counts
                                # + voice-engine status (for alert debugging)

Tuning: edit `config/settings.yaml` — the key knob is `touch_distance_threshold`
(touch closer/farther) and `approach_distance_threshold`. Lower values = the
hand must be closer before the system reacts.

## 15. Example Output

Console (updated once per second):

```text
[TouchGuard AI] Monitoring started. Press 'q' to quit.
[TouchGuard AI] Controls: q=quit p=pause m=mute r=reset t=test voice
[TouchGuard AI] STATUS: SAFE | Faces: 1 | Hands: 0 | FACE TOUCH COUNT: 0 | Hand -> Face Distance: 412 px
[TouchGuard AI] STATUS: APPROACHING | Faces: 1 | Hands: 2 | FACE TOUCH COUNT: 0 | Hand -> Face Distance: 145 px
[TouchGuard AI] STATUS: TOUCHING | Faces: 1 | Hands: 2 | FACE TOUCH COUNT: 1 | Hand -> Face Distance: 35 px
[TouchGuard AI] STATUS: SAFE | Faces: 1 | Hands: 0 | FACE TOUCH COUNT: 1 | Hand -> Face Distance: 430 px
[TouchGuard AI] STATUS: TOUCHING | Faces: 1 | Hands: 2 | FACE TOUCH COUNT: 2 | Hand -> Face Distance: 30 px
```

On-screen (OpenCV window):

```text
Status: ⚠ FACE TOUCH DETECTED
FACE TOUCH COUNT: 4
Hand -> Face: 35 px
```

With: green face box, white forehead/nose/cheeks/mouth-chin markers, cyan hand
skeleton, red fingertips, a yellow distance line from the nearest fingertip to
the nearest face region, FPS, and a flashing banner when a touch is detected.
Every separate touch — and nothing while the hand stays on the face — speaks:

> “Warning. You touched your face. Touch number 1, 2, 3… each new touch.”

## 16. Testing

```bash
pip install -r requirements-dev.txt
pytest -v
```

The suite is entirely webcam-free — heavy components (MediaPipe, TTS, webcam)
are mocked/stubbed, and the touch logic is tested directly. Coverage includes
the distance zones (SAFE / APPROACHING / TOUCH), debounce, release-margin
re-arming, one-warning-per-event, touch counting, wrist toggling, pixel-space
landmark conversion, model-missing handling, voice cooldown/fallback, and webcam
failure handling.
## 17. Future Enhancements

- Face-landmark blending for more precise face-region detection.
- Skin-region or mask classification to reduce false positives.
- Touch analytics such as touches per minute and daily reports.
- CSV/JSON-based historical touch logs.
- Packaged desktop executable using PyInstaller.
- Multi-camera support.
- Multi-person support.
- Automatic distance calibration for different webcam resolutions.
## 18. Limitations

- **Single user, single camera** — designed for one face in front of the webcam.
- **Proximity heuristic** — "touching" is estimated geometrically; brief
  far-side waves or a hand near the face but not touching may occasionally
  trigger.
- **Lighting & occlusion** — Haar and MediaPipe degrade in poor lighting,
  extreme angles, or when the hand covers the face.
- **Not medical-grade** — treat results as a behavioural reminder aid.
- **Voice platform notes** — pyttsx3 needs the OS speech engine (present by
  default on Windows). Otherwise a beep is used.
## 19. Author

**TouchGuard AI Team** — built for learning and demonstrating real-time computer-vision / ML concepts
