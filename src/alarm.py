"""
alarm.py
--------
Manages audible alarm playback when drowsiness is detected.

`playsound` is a blocking call -- invoking it directly from the main
video loop would freeze frame capture and rendering for the entire
duration of the alarm sound. This module instead runs playback on a
dedicated daemon thread, guarded by a lock so overlapping calls can
never spawn multiple simultaneous alarm sounds, and enforces a cooldown
period so a continuously-drowsy driver doesn't get spammed with
overlapping alarm restarts.

All playback failures (missing audio backend, unsupported file,
missing output device, etc.) are caught and logged -- an alarm failure
must NEVER crash the detection loop, since the visual "DROWSY" UI
warning remains fully functional even if audio fails.
"""

import threading
import time
from typing import Optional

from config import settings


class AlarmManager:
    """Thread-safe, non-blocking, cooldown-protected alarm player."""

    def __init__(
        self,
        sound_path: str = settings.ALARM_SOUND_PATH,
        cooldown_seconds: float = settings.ALARM_COOLDOWN_SECONDS,
        enabled: bool = settings.ALARM_ENABLED,
    ) -> None:
        self._sound_path = sound_path
        self._cooldown_seconds = cooldown_seconds
        self._enabled = enabled

        self._lock = threading.Lock()
        self._is_playing = False
        self._last_triggered_at: Optional[float] = None

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing

    def trigger(self) -> bool:
        """
        Request that the alarm sound be played, subject to the
        configured cooldown and the "already playing" guard.

        Returns:
            True if a new playback thread was started, False if the
            request was suppressed (disabled, cooling down, or already
            playing).
        """
        if not self._enabled:
            return False

        now = time.time()

        with self._lock:
            if self._is_playing:
                return False

            if self._last_triggered_at is not None:
                elapsed_since_last = now - self._last_triggered_at
                if elapsed_since_last < self._cooldown_seconds:
                    return False

            self._is_playing = True
            self._last_triggered_at = now

        thread = threading.Thread(target=self._play_sound_safely, daemon=True)
        thread.start()
        return True

    def _play_sound_safely(self) -> None:
        """Runs on a background thread; never allowed to raise."""
        try:
            from playsound import playsound

            playsound(self._sound_path)
        except Exception as exc:  # pragma: no cover - platform/audio dependent
            print(
                f"[WARNING] Alarm playback failed ({exc}). "
                f"Verify that '{self._sound_path}' exists and that a working "
                f"audio backend is available on this system. The visual "
                f"drowsiness warning will continue to function normally."
            )
        finally:
            with self._lock:
                self._is_playing = False

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable alarm playback at runtime."""
        self._enabled = enabled
