"""
eye_tracker.py
---------------
Pure-mathematics module responsible for computing the Eye Aspect
Ratio (EAR) from a set of 6 eye-contour landmark points, following the
formula introduced by Soukupova & Cech (2016), "Real-Time Eye Blink
Detection using Facial Landmarks".

Given 6 points P1..P6 traced around one eye (P1 and P4 at the
horizontal corners; P2, P3 above the pupil; P5, P6 below it):

              ||P2 - P6|| + ||P3 - P5||
    EAR = ---------------------------------
                    2 * ||P1 - P4||

The EAR stays roughly constant while an eye is open and drops sharply
toward zero as the eye closes, making it a simple yet effective
drowsiness signal. This module contains no OpenCV/MediaPipe/dlib code,
so it can be unit tested with plain coordinate tuples.
"""

from collections import deque
from typing import Deque, List, Tuple

from scipy.spatial import distance as scipy_distance

Point = Tuple[int, int]


class EyeAspectRatioCalculator:
    """Computes single-eye and combined-eye Eye Aspect Ratio values."""

    @staticmethod
    def compute_ear(eye_points: List[Point]) -> float:
        """
        Args:
            eye_points: Exactly 6 (x, y) points in P1..P6 order.

        Returns:
            The Eye Aspect Ratio as a float. Typically ~0.25-0.35 for
            an open eye and drops below ~0.2 for a closed eye, though
            exact values vary by face geometry and camera angle.
        """
        if len(eye_points) != 6:
            raise ValueError(f"Expected 6 eye landmark points, received {len(eye_points)}.")

        p1, p2, p3, p4, p5, p6 = eye_points

        vertical_1 = scipy_distance.euclidean(p2, p6)
        vertical_2 = scipy_distance.euclidean(p3, p5)
        horizontal = scipy_distance.euclidean(p1, p4)

        if horizontal <= 1e-6:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    @classmethod
    def compute_average_ear(cls, left_eye: List[Point], right_eye: List[Point]) -> float:
        """Return the mean EAR across both eyes for a more stable signal."""
        left_ear = cls.compute_ear(left_eye)
        right_ear = cls.compute_ear(right_eye)
        return (left_ear + right_ear) / 2.0


class EARSmoother:
    """
    Applies a simple moving-average filter over the most recent EAR
    readings to reduce landmark-jitter noise before the smoothed value
    is compared against the drowsiness threshold.
    """

    def __init__(self, window_size: int) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        self._window: Deque[float] = deque(maxlen=window_size)

    def update(self, raw_ear: float) -> float:
        """Feed in the latest raw EAR reading; get back the smoothed value."""
        self._window.append(raw_ear)
        return sum(self._window) / len(self._window)

    def reset(self) -> None:
        """Clear all accumulated history (e.g. when the face is lost)."""
        self._window.clear()
