"""Backward-compatible entry point for TouchGuard AI.

``python main.py`` runs the exact same desktop application as ``python
app.py``. New users should prefer ``python app.py``.

Controls: ``q`` quit · ``p`` pause/resume · ``m`` mute/unmute voice · ``r`` reset · ``t`` test voice warning
"""

from __future__ import annotations

import sys

from app import main, run_app  # noqa: F401  - re-exported for compatibility

__all__ = ["main", "run_app"]


if __name__ == "__main__":
    sys.exit(main())