#!/usr/bin/env python3
"""
AI Drowsiness Detection System
================================
Main application entry point.

A real-time computer vision safety application that monitors a
driver's (or any user's) eyes via webcam, computes the Eye Aspect
Ratio (EAR) from facial landmarks, and raises an audible + visual
alarm when sustained eye closure indicates drowsiness. Built on a
dual-backend detection layer (MediaPipe by default, dlib optionally)
so the project runs immediately after installing dependencies while
still honoring the classic dlib-based EAR research approach.

Author:  Abid Ali
Project: AI & Machine Learning Diploma Portfolio
License: MIT (see LICENSE file)

Usage:
    python main.py

Keyboard Shortcuts:
    Q / ESC  -- Exit the application
    S        -- Save a timestamped snapshot of the current frame
"""

import datetime
import os
import sys
import time

import cv2

from config import settings
from src import (
    AlarmManager,
    DrowsinessDetector,
    DrowsinessStatus,
    EARSmoother,
    EyeAspectRatioCalculator,
    FPSCounter,
    create_face_detector,
)
from src import utils


class CameraNotAvailableError(Exception):
    """Raised when the webcam cannot be opened after all retry attempts."""


class DrowsinessDetectionApp:
    """
    Top-level application controller.

    Owns the video capture device and coordinates the face/eye
    detection, EAR computation, drowsiness state machine, alarm
    manager, and UI-rendering pipeline on every frame of the main loop.
    """

    def __init__(self) -> None:
        self._capture = self._open_camera()
        self._face_detector = create_face_detector()
        self._ear_calculator = EyeAspectRatioCalculator()
        self._ear_smoother = EARSmoother(window_size=settings.EAR_SMOOTHING_WINDOW)
        self._drowsiness_detector = DrowsinessDetector()
        self._alarm_manager = AlarmManager()
        self._fps_counter = FPSCounter()

        self._webcam_connected = True
        self._snapshot_dir = os.path.join(os.getcwd(), "snapshots")
        self._last_confidence = 0.0

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------
    @staticmethod
    def _open_camera() -> cv2.VideoCapture:
        """Attempt to open the configured webcam with bounded retries."""
        last_error = None
        for _ in range(settings.CAMERA_RECONNECT_ATTEMPTS):
            try:
                capture = cv2.VideoCapture(settings.CAMERA_INDEX)
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
                    for _ in range(settings.CAMERA_WARMUP_FRAMES):
                        capture.read()
                    return capture
                capture.release()
            except Exception as exc:  # pragma: no cover - hardware dependent
                last_error = exc
            time.sleep(0.5)

        raise CameraNotAvailableError(
            f"Could not access webcam at index {settings.CAMERA_INDEX} "
            f"after {settings.CAMERA_RECONNECT_ATTEMPTS} attempts. "
            f"Ensure no other application is using the camera and that "
            f"OS-level camera permissions are granted."
            + (f" Last error: {last_error}" if last_error else "")
        )

    # ------------------------------------------------------------------
    # UI rendering
    # ------------------------------------------------------------------
    def _draw_top_bar(self, frame) -> None:
        height, width = frame.shape[:2]
        utils.draw_translucent_rect(
            frame, (0, 0), (width, settings.TOP_BAR_HEIGHT), settings.COLOR_PANEL_BG, alpha=0.65
        )
        utils.draw_text(
            frame,
            "AI DROWSINESS DETECTION SYSTEM",
            (20, 35),
            scale=settings.FONT_SCALE_MEDIUM,
            color=settings.COLOR_SECONDARY,
            thickness=2,
        )
        utils.draw_text(
            frame,
            f"Backend: {type(self._face_detector).__name__}",
            (20, 68),
            scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_TEXT_LIGHT,
            thickness=1,
        )

        fps_text = f"FPS: {self._fps_counter.fps:5.1f}"
        text_w, _ = utils.get_text_size(fps_text, scale=settings.FONT_SCALE_MEDIUM, thickness=2)
        utils.draw_text(
            frame, fps_text, (width - text_w - 25, 35), scale=settings.FONT_SCALE_MEDIUM,
            color=settings.COLOR_SUCCESS, thickness=2,
        )

        status_text = "WEBCAM: LIVE" if self._webcam_connected else "WEBCAM: LOST"
        status_color = settings.COLOR_SUCCESS if self._webcam_connected else settings.COLOR_DANGER
        status_w, _ = utils.get_text_size(status_text, scale=settings.FONT_SCALE_SMALL, thickness=1)
        utils.draw_text(
            frame, status_text, (width - status_w - 25, 68), scale=settings.FONT_SCALE_SMALL,
            color=status_color, thickness=1,
        )
        cv2.circle(frame, (width - status_w - 40, 62), 5, status_color, cv2.FILLED)

    def _draw_status_panel(self, frame, drowsiness_state, confidence: float) -> None:
        height, width = frame.shape[:2]
        panel_x1 = 20
        panel_y1 = settings.TOP_BAR_HEIGHT + 20
        panel_x2 = panel_x1 + settings.STATUS_PANEL_WIDTH
        panel_y2 = panel_y1 + settings.STATUS_PANEL_HEIGHT

        is_drowsy = drowsiness_state.status == DrowsinessStatus.DROWSY
        accent = settings.COLOR_DANGER if is_drowsy else settings.COLOR_SUCCESS

        utils.draw_rounded_panel(
            frame, (panel_x1, panel_y1), (panel_x2, panel_y2), settings.COLOR_PANEL_BG,
            alpha=0.7, border_color=accent, border_thickness=2,
        )

        utils.draw_text(
            frame, "STATUS", (panel_x1 + 15, panel_y1 + 26), scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_TEXT_LIGHT, thickness=1,
        )
        utils.draw_text(
            frame, drowsiness_state.status.value.upper(), (panel_x1 + 15, panel_y1 + 65),
            scale=settings.FONT_SCALE_LARGE, color=accent, thickness=3,
        )

        utils.draw_text(
            frame, f"EAR: {drowsiness_state.current_ear:.3f}  (Threshold: {self._drowsiness_detector.ear_threshold:.2f})",
            (panel_x1 + 15, panel_y1 + 92), scale=settings.FONT_SCALE_SMALL, color=settings.COLOR_TEXT_LIGHT, thickness=1,
        )
        utils.draw_text(
            frame, f"Blinks: {drowsiness_state.blink_count}",
            (panel_x1 + 15, panel_y1 + 114), scale=settings.FONT_SCALE_SMALL, color=settings.COLOR_TEXT_LIGHT, thickness=1,
        )

        utils.draw_confidence_bar(
            frame, (panel_x1 + 15, panel_y1 + 124), width=settings.STATUS_PANEL_WIDTH - 30, height=10,
            confidence=drowsiness_state.drowsy_progress,
            color=settings.COLOR_DANGER if drowsiness_state.drowsy_progress > 0.5 else settings.COLOR_WARNING,
        )

        conf_panel_y1 = panel_y2 + 15
        conf_panel_y2 = conf_panel_y1 + 60
        utils.draw_rounded_panel(
            frame, (panel_x1, conf_panel_y1), (panel_x2, conf_panel_y2), settings.COLOR_PANEL_BG,
            alpha=0.65, border_color=settings.COLOR_SECONDARY, border_thickness=2,
        )
        utils.draw_text(
            frame, "Tracking Confidence", (panel_x1 + 15, conf_panel_y1 + 24), scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_TEXT_LIGHT, thickness=1,
        )
        utils.draw_confidence_bar(
            frame, (panel_x1 + 15, conf_panel_y1 + 34), width=settings.STATUS_PANEL_WIDTH - 30, height=10,
            confidence=confidence, color=settings.COLOR_SUCCESS,
        )

    def _draw_drowsy_warning_banner(self, frame) -> None:
        height, width = frame.shape[:2]
        # A pulsing-style banner using the current second's parity for a
        # simple, dependency-free "flashing alert" effect.
        if int(time.time() * 2) % 2 == 0:
            utils.draw_translucent_rect(frame, (0, height - 130), (width, height - 40), settings.COLOR_DANGER, alpha=0.55)
            utils.draw_centered_text(
                frame, "!!! DROWSINESS ALERT -- WAKE UP !!!", height - 85,
                scale=1.1, color=settings.COLOR_TEXT_LIGHT, thickness=3,
            )

    def _draw_footer(self, frame) -> None:
        height, width = frame.shape[:2]
        footer_text = "Press [Q] or [ESC] to Exit   |   Press [S] to Save Snapshot"
        text_w, text_h = utils.get_text_size(footer_text, scale=settings.FONT_SCALE_SMALL, thickness=1)
        utils.draw_translucent_rect(
            frame, (0, height - text_h - 24), (width, height), settings.COLOR_PANEL_BG, alpha=0.55
        )
        utils.draw_text(
            frame, footer_text, ((width - text_w) // 2, height - 15), scale=settings.FONT_SCALE_SMALL,
            color=settings.COLOR_TEXT_LIGHT, thickness=1, shadow=False,
        )

    def _save_snapshot(self, frame) -> None:
        os.makedirs(self._snapshot_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self._snapshot_dir, f"snapshot_{timestamp}.png")
        cv2.imwrite(filepath, frame)
        print(f"[INFO] Snapshot saved to: {filepath}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        print("[INFO] AI Drowsiness Detection System starting...")
        print(f"[INFO] Detection backend: {type(self._face_detector).__name__}")
        print("[INFO] Press 'Q' or 'ESC' to exit, 'S' to save a snapshot.")

        cv2.namedWindow(settings.WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            while True:
                success, frame = self._capture.read()

                if not success or frame is None:
                    self._webcam_connected = False
                    print("[WARNING] Failed to read frame from webcam. Retrying...")
                    time.sleep(0.5)
                    continue

                self._webcam_connected = True

                if settings.FLIP_CAMERA_HORIZONTALLY:
                    frame = cv2.flip(frame, 1)

                face_landmarks = self._face_detector.detect(frame)

                if face_landmarks is not None:
                    self._last_confidence = face_landmarks.confidence

                    raw_ear = self._ear_calculator.compute_average_ear(
                        face_landmarks.left_eye, face_landmarks.right_eye
                    )
                    smoothed_ear = self._ear_smoother.update(raw_ear)
                    drowsiness_state = self._drowsiness_detector.update(smoothed_ear)

                    utils.draw_eye_contour(frame, face_landmarks.left_eye)
                    utils.draw_eye_contour(frame, face_landmarks.right_eye)
                    utils.draw_bounding_box(
                        frame, face_landmarks.face_bounding_box, "Face Detected", settings.COLOR_SECONDARY
                    )

                    if drowsiness_state.status == DrowsinessStatus.DROWSY:
                        self._alarm_manager.trigger()
                        self._draw_drowsy_warning_banner(frame)

                else:
                    # No face detected this frame -- reset the in-progress
                    # closure timer (not the blink tally) so a temporary
                    # tracking dropout is never mistaken for eye closure.
                    self._ear_smoother.reset()
                    self._drowsiness_detector.reset()
                    drowsiness_state = self._drowsiness_detector.update(1.0)  # Force an "open-eyes" snapshot
                    self._last_confidence = 0.0
                    utils.draw_centered_text(
                        frame, "No Face Detected", frame.shape[0] // 2, scale=1.0,
                        color=settings.COLOR_WARNING, thickness=2,
                    )

                self._fps_counter.tick()
                self._draw_top_bar(frame)
                self._draw_status_panel(frame, drowsiness_state, self._last_confidence)
                self._draw_footer(frame)

                cv2.imshow(settings.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in settings.EXIT_KEYS:
                    print("[INFO] Exit key pressed. Shutting down...")
                    break
                if key == settings.SNAPSHOT_KEY:
                    self._save_snapshot(frame)

                if cv2.getWindowProperty(settings.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed by user. Shutting down...")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (Ctrl+C). Shutting down...")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Release all hardware and library resources deterministically."""
        self._face_detector.close()
        if self._capture is not None:
            self._capture.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Goodbye!")


def main() -> int:
    """Application bootstrap with top-level error handling."""
    try:
        app = DrowsinessDetectionApp()
        app.run()
        return 0
    except CameraNotAvailableError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - top-level safety net
        print(f"[FATAL] An unexpected error occurred: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
