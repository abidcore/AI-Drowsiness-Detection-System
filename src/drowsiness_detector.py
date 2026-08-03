"""
drowsiness_detector.py
------------------------
Pure-logic state machine that converts a stream of Eye Aspect Ratio
(EAR) readings into a drowsiness classification over time. Contains no
OpenCV/UI code, so the state machine can be exercised and unit tested
completely independently of the camera pipeline.

Design notes:
    * All timing is wall-clock based (via injectable timestamps),
      NOT frame-count based, so behavior is identical whether the
      host machine renders at 15 FPS or 60 FPS.
    * A short eye closure is classified as a normal BLINK and simply
      tallied; only a *sustained* closure beyond
      `settings.DROWSY_DURATION_THRESHOLD_SECONDS` escalates the
      status to DROWSY. This distinction is what separates this
      implementation from naive "N consecutive frames closed"
      tutorials, which conflate blinking with drowsiness.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import settings


class DrowsinessStatus(Enum):
    """The two headline states surfaced to the UI and alarm system."""

    AWAKE = "Awake"
    DROWSY = "Drowsy"


@dataclass
class DrowsinessState:
    """Immutable snapshot of the detector's output for a single update."""

    status: DrowsinessStatus
    current_ear: float
    eyes_closed_duration: float   # Seconds the eyes have been continuously closed (0 if open)
    blink_count: int              # Total normal blinks tallied so far this session
    drowsy_progress: float        # 0.0-1.0 progress toward the drowsy threshold, for a UI progress bar


class DrowsinessDetector:
    """
    Tracks eye-closure duration over time and classifies the driver as
    AWAKE or DROWSY, while separately tallying normal blinks.
    """

    def __init__(
        self,
        ear_threshold: float = settings.EAR_THRESHOLD,
        drowsy_duration_threshold: float = settings.DROWSY_DURATION_THRESHOLD_SECONDS,
        blink_max_duration: float = settings.BLINK_MAX_DURATION_SECONDS,
    ) -> None:
        self._ear_threshold = ear_threshold
        self._drowsy_duration_threshold = drowsy_duration_threshold
        self._blink_max_duration = blink_max_duration

        self._eyes_closed_since: Optional[float] = None
        self._blink_count: int = 0
        self._status: DrowsinessStatus = DrowsinessStatus.AWAKE

    @property
    def ear_threshold(self) -> float:
        return self._ear_threshold

    def set_ear_threshold(self, new_threshold: float) -> None:
        """Allow runtime adjustment of detection sensitivity."""
        if new_threshold <= 0:
            raise ValueError("EAR threshold must be a positive value.")
        self._ear_threshold = new_threshold

    def update(self, ear_value: float, timestamp: Optional[float] = None) -> DrowsinessState:
        """
        Feed in the latest (smoothed) EAR reading and receive the
        current drowsiness classification back.

        Args:
            ear_value: The current (ideally already-smoothed) EAR value.
            timestamp: Optional injectable wall-clock time in seconds
                (via time.time()), primarily for deterministic unit
                testing. Defaults to the current time.

        Returns:
            A DrowsinessState snapshot describing the current status.
        """
        now = timestamp if timestamp is not None else time.time()
        eyes_are_closed = ear_value < self._ear_threshold

        if eyes_are_closed:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = now

            closed_duration = now - self._eyes_closed_since

            if closed_duration >= self._drowsy_duration_threshold:
                self._status = DrowsinessStatus.DROWSY
            else:
                self._status = DrowsinessStatus.AWAKE

        else:
            if self._eyes_closed_since is not None:
                closed_duration = now - self._eyes_closed_since
                if closed_duration < self._blink_max_duration:
                    self._blink_count += 1
                self._eyes_closed_since = None

            closed_duration = 0.0
            self._status = DrowsinessStatus.AWAKE

        progress = min(closed_duration / self._drowsy_duration_threshold, 1.0) if self._drowsy_duration_threshold > 0 else 0.0

        return DrowsinessState(
            status=self._status,
            current_ear=ear_value,
            eyes_closed_duration=closed_duration,
            blink_count=self._blink_count,
            drowsy_progress=progress,
        )

    def reset(self) -> None:
        """
        Clear the in-progress eye-closure timer without penalizing the
        blink tally. Intended to be called when the face/eyes are lost
        from view entirely (as opposed to genuinely closing), so a
        temporary tracking dropout is never mistaken for a closure.
        """
        self._eyes_closed_since = None
        self._status = DrowsinessStatus.AWAKE

    def reset_session(self) -> None:
        """Fully reset the detector, including the blink counter."""
        self._eyes_closed_since = None
        self._blink_count = 0
        self._status = DrowsinessStatus.AWAKE
