"""Windows voice warning service for TouchGuard AI."""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from typing import Any, Callable

from touchguard.config import VoiceConfig

logger = logging.getLogger(__name__)


class VoiceWarningService:
    """Voice warning service using Windows PowerShell SpeechSynthesizer."""

    def __init__(
        self,
        voice_config: VoiceConfig | None = None,
        engine_factory: Callable[[], Any] | None = None,
        start_worker: bool = True,
    ):
        self._config = voice_config or VoiceConfig()

        self._enabled = bool(self._config.enabled)
        self._min_interval = max(
            0.0,
            float(self._config.min_interval_seconds),
        )

        self._message = self._config.main_warning

        self._mode = "windows-speech"
        self._init_error = None

        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = None

        self._lock = threading.Lock()
        self._last_warn_time = 0.0

        self.warning_count = 0

        if start_worker:
            self._start_worker()

        logger.info("Windows speech warning service initialized.")

    def _start_worker(self):
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="touchguard-voice",
            daemon=True,
        )
        self._worker.start()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if job is None:
                break

            self._deliver(job)

    def _deliver(self, job):
        kind, message = job

        if kind != "say":
            return

        try:
            # Use a temporary PowerShell script so that quotation
            # marks in the warning message cannot break the command.
            ps_script = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Volume = 100
$s.Rate = 0
$s.Speak("{str(message).replace('"', '`"')}")
$s.Dispose()
"""

            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    ps_script,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

            logger.info("Spoken warning: %s", message)

        except Exception as exc:
            logger.error("Windows speech failed: %s", exc)
            self._beep()

    @staticmethod
    def _beep():
        try:
            import winsound
            winsound.Beep(880, 250)
        except Exception:
            pass

    @property
    def enabled(self):
        return self._enabled

    @property
    def available(self):
        return True

    @property
    def mode(self):
        return self._mode

    @property
    def cooldown_remaining(self):
        now = time.monotonic()

        remaining = self._min_interval - (
            now - self._last_warn_time
        )

        return max(0.0, remaining)

    def announce(self, message=None):
        if not self._enabled:
            return False

        now = time.monotonic()

        with self._lock:
            if now - self._last_warn_time < self._min_interval:
                return False

            self._last_warn_time = now
            self.warning_count += 1

        payload = message or self._message

        self._queue.put(("say", payload))

        return True

    def set_enabled(self, enabled: bool):
        self._enabled = bool(enabled)

    def reset(self):
        with self._lock:
            self._last_warn_time = 0.0
            self.warning_count = 0

    def flush(self):
        time.sleep(0.1)

    def stop(self):
        self._stop_event.set()

        try:
            self._queue.put(None)
        except Exception:
            pass

        if self._worker is not None:
            self._worker.join(timeout=2.0)