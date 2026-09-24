# Models

Hand tracking uses the MediaPipe **Hand Landmarker** (Tasks API). The model is
downloaded automatically on the first run when missing.

| File | Purpose | Source |
| --- | --- | --- |
| `hand_landmarker.task` | 21-point hand landmark detection (float16) | [MediaPipe models](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) |

Face detection model lives in `assets/models/` (Haar cascade, also auto-downloaded).

To pre-download models manually:

```bash
python scripts/download_models.py
```

Reference: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task