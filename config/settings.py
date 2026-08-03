"""
config/settings.py
-------------------
Centralized, single-source-of-truth configuration for the AI
Drowsiness Detection System.

Every tunable constant -- camera parameters, detection-backend
selection, EAR sensitivity thresholds, alarm behavior, and UI layout
-- lives here rather than being scattered as "magic numbers"
throughout the codebase. This is also where a college evaluator or
end user should look first to *adjust detection sensitivity*, per the
project's design requirement.
"""

import os

import cv2

# --------------------------------------------------------------------------
# Project Paths
# --------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR: str = os.path.join(BASE_DIR, "models")
ASSETS_DIR: str = os.path.join(BASE_DIR, "assets")

DLIB_LANDMARK_MODEL_PATH: str = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
ALARM_SOUND_PATH: str = os.path.join(ASSETS_DIR, "alarm.wav")

# Official dlib model download source, referenced in error messages and
# README instructions when the .dat file is not present locally.
DLIB_MODEL_DOWNLOAD_URL: str = (
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
)

# --------------------------------------------------------------------------
# Camera / Capture Settings
# --------------------------------------------------------------------------
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
FLIP_CAMERA_HORIZONTALLY: bool = True
CAMERA_WARMUP_FRAMES: int = 5
CAMERA_RECONNECT_ATTEMPTS: int = 3

# --------------------------------------------------------------------------
# Face / Landmark Detection Backend
# --------------------------------------------------------------------------
# The system supports two interchangeable detection backends behind a
# single common interface (see src/face_detector.py):
#
#   "mediapipe" -- Google's MediaPipe Face Mesh. Requires no external
#                  model download and works immediately after
#                  `pip install -r requirements.txt`. Used as the
#                  DEFAULT backend so the project runs out-of-the-box.
#
#   "dlib"      -- Classic HOG face detector + 68-point landmark
#                  predictor, exactly as used in the original academic
#                  EAR-based drowsiness-detection research. Requires a
#                  one-time download of a ~100 MB model file (see
#                  README.md "Enabling the dlib Backend" section, or
#                  let the app auto-download it on first run).
#
# Both backends produce the same standardized output (6 eye landmark
# points per eye), so drowsiness_detector.py and eye_tracker.py are
# completely backend-agnostic.
DETECTION_BACKEND: str = "mediapipe"  # "mediapipe" or "dlib"

# If the dlib backend is selected but its model file is missing, allow
# the application to transparently fall back to MediaPipe instead of
# crashing -- with a clear console warning explaining what happened.
FALLBACK_TO_MEDIAPIPE_IF_DLIB_UNAVAILABLE: bool = True

# Attempt to automatically download the dlib model file on first use
# if it is missing and this flag is enabled (requires an internet
# connection). If False, a clear manual-download error is raised.
AUTO_DOWNLOAD_DLIB_MODEL: bool = True

MIN_DETECTION_CONFIDENCE: float = 0.6
MIN_TRACKING_CONFIDENCE: float = 0.6

# --------------------------------------------------------------------------
# Eye Aspect Ratio (EAR) Sensitivity Settings
# --------------------------------------------------------------------------
# These are the primary "detection sensitivity" knobs referenced in the
# project's feature list. Lowering EAR_THRESHOLD makes the system less
# sensitive (requires eyes to be more closed before triggering);
# raising it makes the system more sensitive.
EAR_THRESHOLD: float = 0.25

# How long (in seconds) the eyes must remain continuously below the
# EAR threshold before the system escalates to a "Drowsy" alarm state.
# Time-based (not frame-count-based) so behavior is consistent
# regardless of the machine's actual FPS.
DROWSY_DURATION_THRESHOLD_SECONDS: float = 1.5

# A closure shorter than this is classified as a normal blink rather
# than a drowsiness event, and is simply tallied in the blink counter.
BLINK_MAX_DURATION_SECONDS: float = 0.4

# Smoothing window (in samples) for the EAR signal itself, reducing
# landmark-jitter noise before it ever reaches the threshold comparison.
EAR_SMOOTHING_WINDOW: int = 5

# --------------------------------------------------------------------------
# Alarm Settings
# --------------------------------------------------------------------------
ALARM_ENABLED: bool = True
ALARM_COOLDOWN_SECONDS: float = 3.0  # Minimum gap between alarm re-triggers

# --------------------------------------------------------------------------
# FPS Counter Settings
# --------------------------------------------------------------------------
FPS_SMOOTHING_FACTOR: float = 0.9

# --------------------------------------------------------------------------
# Window / Display Settings
# --------------------------------------------------------------------------
WINDOW_NAME: str = "AI Drowsiness Detection System | Abid Ali"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LARGE: float = 1.3
FONT_SCALE_MEDIUM: float = 0.75
FONT_SCALE_SMALL: float = 0.58
FONT_THICKNESS: int = 2

# --------------------------------------------------------------------------
# Color Palette (BGR format, as used by OpenCV)
# --------------------------------------------------------------------------
COLOR_PRIMARY = (255, 130, 0)
COLOR_SECONDARY = (0, 210, 255)
COLOR_SUCCESS = (80, 220, 100)      # Awake / connected
COLOR_DANGER = (60, 60, 235)        # Drowsy / disconnected
COLOR_WARNING = (0, 165, 255)       # Blink / caution
COLOR_TEXT_LIGHT = (245, 245, 245)
COLOR_TEXT_DARK = (25, 25, 25)
COLOR_PANEL_BG = (35, 35, 35)
COLOR_EYE_CONTOUR = (0, 210, 255)

# --------------------------------------------------------------------------
# UI Layout Settings
# --------------------------------------------------------------------------
PANEL_OPACITY: float = 0.6
TOP_BAR_HEIGHT: int = 90
CORNER_RADIUS: int = 14
STATUS_PANEL_WIDTH: int = 320
STATUS_PANEL_HEIGHT: int = 150

# --------------------------------------------------------------------------
# Keyboard Shortcuts
# --------------------------------------------------------------------------
EXIT_KEYS = {ord("q"), ord("Q"), 27}  # 'q', 'Q', and ESC key code
SNAPSHOT_KEY = ord("s")
