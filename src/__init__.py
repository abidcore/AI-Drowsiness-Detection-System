"""
src
---
Core package for the AI Drowsiness Detection System.

Modules:
    face_detector       -- Dual-backend (MediaPipe/dlib) face and eye landmark detection
    eye_tracker         -- Eye Aspect Ratio (EAR) computation and smoothing
    drowsiness_detector -- Time-based drowsiness state machine and blink discrimination
    alarm               -- Threaded, cooldown-protected audible alarm playback
    fps                 -- Lightweight, smoothed frames-per-second counter
    utils               -- Reusable OpenCV drawing/UI helper functions
"""

from .alarm import AlarmManager
from .drowsiness_detector import DrowsinessDetector, DrowsinessState, DrowsinessStatus
from .eye_tracker import EARSmoother, EyeAspectRatioCalculator
from .face_detector import (
    BaseFaceDetector,
    DlibFaceDetector,
    FaceLandmarks,
    MediaPipeFaceDetector,
    create_face_detector,
)
from .fps import FPSCounter

__all__ = [
    "BaseFaceDetector",
    "MediaPipeFaceDetector",
    "DlibFaceDetector",
    "FaceLandmarks",
    "create_face_detector",
    "EyeAspectRatioCalculator",
    "EARSmoother",
    "DrowsinessDetector",
    "DrowsinessState",
    "DrowsinessStatus",
    "AlarmManager",
    "FPSCounter",
]

__version__ = "1.0.0"
