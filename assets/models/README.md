# Models

Face detection is performed with OpenCV Haar cascades. The default cascade is
downloaded automatically (from the official OpenCV repository) on the first
run when missing:

| File | Purpose |
| --- | --- |
| `haarcascade_frontalface_default.xml` | Front-facing face detector |

To pre-download the model manually:

```bash
python scripts/download_models.py
```

Reference:
https://github.com/opencv/opencv/tree/master/data/haarcascades