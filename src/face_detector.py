"""
face_detector.py
------------------
Backend-agnostic face and eye-landmark detection layer.

This module is the architectural heart of the system's flexibility: it
exposes a single, uniform `FaceDetector` interface backed by TWO
interchangeable implementations --

    * `MediaPipeFaceDetector` -- Google's MediaPipe Face Mesh. Requires
      no external model download and works immediately after
      `pip install -r requirements.txt`. Used as the DEFAULT backend.

    * `DlibFaceDetector` -- the classic HOG face detector + 68-point
      landmark predictor used in the original academic EAR-based
      drowsiness-detection research. Requires a one-time model file
      download (handled automatically, see `_ensure_dlib_model()`).

Both backends normalize their output into the same `FaceLandmarks`
dataclass -- six (x, y) eye-contour points per eye, in the exact P1-P6
ordering the Eye Aspect Ratio formula expects -- so every downstream
module (`eye_tracker.py`, `drowsiness_detector.py`) is completely
unaware of which backend produced the data.
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import settings

Point = Tuple[int, int]


@dataclass
class FaceLandmarks:
    """Standardized, backend-agnostic detection result for one face."""

    left_eye: List[Point]           # 6 pixel-space (x, y) points, P1..P6 order
    right_eye: List[Point]          # 6 pixel-space (x, y) points, P1..P6 order
    face_bounding_box: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    confidence: float               # 0.0 - 1.0, meaning documented per backend


class BaseFaceDetector:
    """Common interface every detection backend must implement."""

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarks]:
        raise NotImplementedError

    def close(self) -> None:
        """Release any backend-specific resources. Optional override."""
        return None


# ----------------------------------------------------------------------------
# MediaPipe backend
# ----------------------------------------------------------------------------

# MediaPipe Face Mesh landmark indices selected to approximate dlib's
# classic 6-point eye contour (corner, corner, and two lid points top
# and bottom), preserving the exact P1..P6 ordering the EAR formula
# relies on: P1=left corner, P2=top-left, P3=top-right, P4=right
# corner, P5=bottom-right, P6=bottom-left.
_MEDIAPIPE_RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
_MEDIAPIPE_LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]


class MediaPipeFaceDetector(BaseFaceDetector):
    """
    Face and eye-landmark detector backed by MediaPipe Face Mesh.

    Because MediaPipe's Python API does not expose a raw per-frame
    face-detection probability for Face Mesh, this backend reports a
    *tracking-stability* confidence: an exponential moving average of
    "was a face found this frame?" over recent frames. This is an
    honest, clearly-documented substitute for a true model confidence
    score, and is labeled as "Tracking Stability" in the UI rather than
    "Detection Confidence" to avoid overstating what it measures.
    """

    def __init__(
        self,
        min_detection_confidence: float = settings.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = settings.MIN_TRACKING_CONFIDENCE,
    ) -> None:
        import mediapipe as mp

        self._mp_face_mesh = mp.solutions.face_mesh
        try:
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Failed to initialize MediaPipe Face Mesh: {exc}") from exc

        self._stability_ema: float = 0.0
        self._stability_alpha: float = 0.85

    def _update_stability(self, detected: bool) -> float:
        sample = 1.0 if detected else 0.0
        self._stability_ema = (
            self._stability_alpha * self._stability_ema + (1 - self._stability_alpha) * sample
        )
        return self._stability_ema

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarks]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            self._update_stability(False)
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        def to_pixel(idx: int) -> Point:
            lm = landmarks[idx]
            return int(lm.x * width), int(lm.y * height)

        left_eye = [to_pixel(i) for i in _MEDIAPIPE_LEFT_EYE_IDX]
        right_eye = [to_pixel(i) for i in _MEDIAPIPE_RIGHT_EYE_IDX]

        all_xs = [lm.x * width for lm in landmarks]
        all_ys = [lm.y * height for lm in landmarks]
        face_bbox = (
            max(int(min(all_xs)), 0),
            max(int(min(all_ys)), 0),
            min(int(max(all_xs)), width),
            min(int(max(all_ys)), height),
        )

        confidence = self._update_stability(True)

        return FaceLandmarks(
            left_eye=left_eye,
            right_eye=right_eye,
            face_bounding_box=face_bbox,
            confidence=confidence,
        )

    def close(self) -> None:
        self._face_mesh.close()


# ----------------------------------------------------------------------------
# dlib backend
# ----------------------------------------------------------------------------

# Note: exact eye-landmark index ranges (36-41 for the right eye,
# 42-47 for the left eye in the 68-point iBUG model) are resolved at
# runtime via imutils.face_utils.FACIAL_LANDMARKS_68_IDXS inside
# DlibFaceDetector.__init__, rather than hard-coded here.


class DlibModelUnavailableError(Exception):
    """Raised when the dlib 68-point landmark model cannot be located or fetched."""


def _ensure_dlib_model() -> str:
    """
    Guarantee the dlib 68-point landmark model file is present on disk,
    downloading and decompressing it automatically if permitted by
    settings.AUTO_DOWNLOAD_DLIB_MODEL and possible.

    Returns:
        The absolute path to the usable .dat model file.

    Raises:
        DlibModelUnavailableError: if the file is missing and cannot
            (or should not) be auto-downloaded.
    """
    model_path = settings.DLIB_LANDMARK_MODEL_PATH

    if os.path.isfile(model_path):
        return model_path

    if not settings.AUTO_DOWNLOAD_DLIB_MODEL:
        raise DlibModelUnavailableError(
            f"dlib landmark model not found at '{model_path}'. "
            f"Download it manually from {settings.DLIB_MODEL_DOWNLOAD_URL}, "
            f"decompress the .bz2 archive, and place the .dat file in the "
            f"'models/' directory. See README.md 'Enabling the dlib Backend'."
        )

    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    compressed_path = model_path + ".bz2"

    try:
        import bz2

        print("[INFO] dlib landmark model not found -- downloading it now (~64 MB)...")
        urllib.request.urlretrieve(settings.DLIB_MODEL_DOWNLOAD_URL, compressed_path)

        print("[INFO] Decompressing dlib landmark model...")
        with bz2.BZ2File(compressed_path) as source, open(model_path, "wb") as destination:
            destination.write(source.read())

        os.remove(compressed_path)
        print(f"[INFO] dlib landmark model ready at: {model_path}")
        return model_path

    except Exception as exc:
        raise DlibModelUnavailableError(
            f"Automatic download of the dlib landmark model failed ({exc}). "
            f"Please download it manually from {settings.DLIB_MODEL_DOWNLOAD_URL}, "
            f"decompress it, and place the .dat file at '{model_path}'."
        ) from exc


class DlibFaceDetector(BaseFaceDetector):
    """
    Face and eye-landmark detector backed by dlib's HOG face detector
    and 68-point landmark predictor.

    Unlike the MediaPipe backend, dlib's HOG+SVM detector exposes a
    genuine detection score via `detector.run(...)`. That raw score is
    an unbounded log-likelihood (typically 0 to ~1.5 for a clear,
    front-facing detection); it is linearly clamped into a 0-1 range
    for display purposes and documented as such.
    """

    _SCORE_CLAMP_MAX = 1.5

    def __init__(self) -> None:
        import dlib
        from imutils import face_utils

        self._dlib = dlib
        self._face_utils = face_utils
        model_path = _ensure_dlib_model()

        try:
            self._detector = dlib.get_frontal_face_detector()
            self._predictor = dlib.shape_predictor(model_path)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Failed to initialize dlib face detector: {exc}") from exc

        # imutils.face_utils exposes the canonical named index ranges
        # for the 68-point iBUG model, avoiding hard-coded "magic"
        # slice bounds and self-documenting exactly which landmark
        # group each slice corresponds to.
        (self._left_start, self._left_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self._right_start, self._right_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarks]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # detector.run() (rather than the plain __call__) additionally
        # returns per-detection confidence scores.
        rectangles, scores, _ = self._detector.run(gray, 0, -1)

        if len(rectangles) == 0:
            return None

        # Select the highest-confidence detection if multiple faces
        # happen to be in frame (the system is designed for one driver).
        best_idx = int(np.argmax(scores))
        rect = rectangles[best_idx]
        raw_score = float(scores[best_idx])
        confidence = max(0.0, min(raw_score / self._SCORE_CLAMP_MAX, 1.0))

        shape = self._predictor(gray, rect)
        # imutils' shape_to_np converts dlib's shape object into a
        # standard (68, 2) NumPy array -- cleaner and more idiomatic
        # than manually looping over shape.part(i) for every point.
        points = self._face_utils.shape_to_np(shape)

        left_eye = [tuple(point) for point in points[self._left_start:self._left_end]]
        right_eye = [tuple(point) for point in points[self._right_start:self._right_end]]

        face_bbox = (
            max(rect.left(), 0),
            max(rect.top(), 0),
            rect.right(),
            rect.bottom(),
        )

        return FaceLandmarks(
            left_eye=left_eye,
            right_eye=right_eye,
            face_bounding_box=face_bbox,
            confidence=confidence,
        )


# ----------------------------------------------------------------------------
# Public factory
# ----------------------------------------------------------------------------


def create_face_detector(backend: Optional[str] = None) -> BaseFaceDetector:
    """
    Instantiate the configured detection backend, with a transparent,
    clearly-logged fallback to MediaPipe if the dlib backend was
    requested but its model is unavailable and fallback is permitted.

    Args:
        backend: Optionally override settings.DETECTION_BACKEND
            ("mediapipe" or "dlib").

    Returns:
        A ready-to-use BaseFaceDetector implementation.
    """
    selected = (backend or settings.DETECTION_BACKEND).lower().strip()

    if selected == "dlib":
        try:
            return DlibFaceDetector()
        except (DlibModelUnavailableError, ImportError) as exc:
            if settings.FALLBACK_TO_MEDIAPIPE_IF_DLIB_UNAVAILABLE:
                print(f"[WARNING] dlib backend unavailable ({exc}). Falling back to MediaPipe.")
                return MediaPipeFaceDetector()
            raise

    if selected != "mediapipe":
        print(f"[WARNING] Unknown DETECTION_BACKEND '{selected}' -- defaulting to MediaPipe.")

    return MediaPipeFaceDetector()
