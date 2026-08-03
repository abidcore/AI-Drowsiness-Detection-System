"""
fps.py
------
A lightweight, self-contained frames-per-second counter using an
exponential moving average (EMA) for a stable, non-jittery readout.

A naive "1 / delta_time" FPS calculation fluctuates wildly frame to
frame; the EMA approach used here produces a smooth, readable value
suitable for an on-screen performance indicator.
"""

import time

from config import settings


class FPSCounter:
    """Tracks and smooths the application's real-time frame rate."""

    def __init__(self, smoothing_factor: float = settings.FPS_SMOOTHING_FACTOR) -> None:
        if not 0.0 < smoothing_factor < 1.0:
            raise ValueError("smoothing_factor must be between 0 and 1 (exclusive).")

        self._smoothing_factor = smoothing_factor
        self._previous_timestamp: float = time.perf_counter()
        self._smoothed_fps: float = 0.0

    def tick(self) -> float:
        """
        Call exactly once per rendered frame.

        Returns:
            The current smoothed FPS value.
        """
        current_timestamp = time.perf_counter()
        elapsed = current_timestamp - self._previous_timestamp
        self._previous_timestamp = current_timestamp

        if elapsed <= 0:
            return self._smoothed_fps

        instantaneous_fps = 1.0 / elapsed

        if self._smoothed_fps == 0.0:
            self._smoothed_fps = instantaneous_fps
        else:
            self._smoothed_fps = (
                self._smoothing_factor * self._smoothed_fps
                + (1.0 - self._smoothing_factor) * instantaneous_fps
            )

        return self._smoothed_fps

    @property
    def fps(self) -> float:
        """The most recently computed smoothed FPS value."""
        return self._smoothed_fps
