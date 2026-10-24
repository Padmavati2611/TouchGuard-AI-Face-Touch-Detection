
"""Windows voice warning service for TouchGuard AI."""

from __future__ import annotations

import base64
import logging
import os
import queue
import subprocess
import threading
import time

from touchguard.config import VoiceConfig

logger = logging.getLogger(__name__)


class VoiceWarningService:
    """Speak warning messages using Windows PowerShell."""

    def __init__(
        self,
        voice_config: VoiceConfig | None = None,
        engine_factory=None,
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

        logger.info("Windows voice warning service initialized.")

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

            try:
                if job is None:
                    break
                self._deliver(job)
            finally:
                self._queue.task_done()

    def _deliver(self, job):
        kind, message = job

        if kind != "say":
            return

        try:
            powershell = os.path.join(
                os.environ.get("WINDIR", r"C:\Windows"),
                "System32",
                "WindowsPowerShell",
                "v1.0",
                "powershell.exe",
            )

            if not os.path.isfile(powershell):
                raise FileNotFoundError(
                    f"PowerShell not found: {powershell}"
                )

            encoded_message = base64.b64encode(
                str(message).encode("utf-16le")
            ).decode("ascii")

            ps_script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$bytes = [Convert]::FromBase64String('{encoded_message}')
$text = [Text.Encoding]::Unicode.GetString($bytes)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{
    $s.Volume = 100
    $s.Rate = 0
    $s.Speak($text)
}}
finally {{
    $s.Dispose()
}}
"""

            result = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    ps_script,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                creationflags=getattr(
                    subprocess, "CREATE_NO_WINDOW", 0
                ),
                timeout=30,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    result.stderr.strip()
                    or f"PowerShell exited with code {result.returncode}"
                )

            logger.info("Spoken warning: %s", message)

        except Exception:
            logger.exception("Windows speech failed.")
            self._beep()

    @staticmethod
    def _beep():
        try:
            import winsound
            winsound.Beep(880, 250)
        except Exception:
            logger.exception("Warning beep failed.")

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
        remaining = self._min_interval - (
            time.monotonic() - self._last_warn_time
        )
        return max(0.0, remaining)

    def announce(self, message=None):
        if not self._enabled:
            return False

        with self._lock:
            now = time.monotonic()

            if now - self._last_warn_time < self._min_interval:
                return False

            self._last_warn_time = now
            self.warning_count += 1

        self._queue.put(("say", message or self._message))
        return True

    def set_enabled(self, enabled: bool):
        self._enabled = bool(enabled)

    def reset(self):
        with self._lock:
            self._last_warn_time = 0.0
            self.warning_count = 0

    def flush(self):
        self._queue.join()

    def stop(self):
        if self._worker is None:
            self._stop_event.set()
            return

        if self._worker.is_alive():
            self._queue.put(None)
            self._worker.join(timeout=5.0)

        self._stop_event.set()
