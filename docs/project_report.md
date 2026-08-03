# Project Report: AI Drowsiness Detection System

**Author:** Abid Ali
**Program:** AI & Machine Learning Diploma
**Project Type:** Real-Time Computer Vision Safety System
**Repository:** [github.com/abidcore/AI-Drowsiness-Detection-System](https://github.com/abidcore)

---

## 1. Introduction

Driver and operator fatigue is a well-documented contributor to road-traffic accidents and industrial safety incidents worldwide. A significant behavioral indicator of impending drowsiness is prolonged or repeated eye closure, which can be detected non-invasively through computer vision using only a standard camera. The **AI Drowsiness Detection System** implements this idea using the **Eye Aspect Ratio (EAR)** technique introduced by Soukupová and Čech (2016), combined with a time-based state machine that reliably distinguishes normal blinking from genuine, sustained drowsiness, and raises an immediate audible and visual alarm when the latter is detected.

This project was undertaken as part of an AI & Machine Learning Diploma program to demonstrate practical, end-to-end applied computer vision engineering: from landmark-based facial analysis, through signal-processing and state-machine design, to a robust, alarm-integrated, real-time safety application built with professional software engineering practices.

---

## 2. Problem Statement

Typical introductory drowsiness-detection tutorials hard-code a single detection library, use a naive "N consecutive frames below threshold" rule that conflates ordinary blinking with genuine drowsiness, call blocking audio-playback functions directly inside the video loop (freezing the feed while the alarm plays), and provide no graceful handling for missing model files, camera failures, or audio-backend errors. These implementations are not representative of a properly engineered real-time safety system.

**The problem this project addresses is threefold:**

1. **Technical:** Reliably compute a stable, noise-resistant Eye Aspect Ratio signal in real time from a 2D webcam feed, and correctly separate normal blinks from dangerous sustained eye closures using wall-clock timing rather than fragile frame counts.
2. **Engineering:** Support the classic dlib-based academic pipeline the EAR technique was originally validated with, while ALSO providing a zero-setup, dependency-light detection path so the project runs immediately after installation — without forcing a large external model download on every user.
3. **Reliability:** Ensure that alarm playback never blocks or freezes video capture/rendering, and that any single point of failure (camera, model file, audio backend) degrades gracefully rather than crashing the entire application.

---

## 3. Objectives

1. Achieve real-time (near 30 FPS) face and eye-landmark tracking on a standard consumer webcam.
2. Implement the Eye Aspect Ratio formula precisely, using SciPy's Euclidean distance function, matching the original research definition.
3. Design a time-based (not frame-count-based) state machine that correctly classifies short closures as blinks and only escalates sustained closures to a "Drowsy" alarm state.
4. Support two interchangeable, architecturally decoupled detection backends (MediaPipe and dlib) behind a single common interface.
5. Implement non-blocking, cooldown-protected audible alarm playback that never freezes the main video loop.
6. Make detection sensitivity (EAR threshold, drowsy-duration threshold) easily adjustable from a single configuration file.
7. Handle camera failures, missing model files, and alarm-playback errors gracefully, with clear diagnostic messaging.
8. Build the system using clean, modular, PEP8-compliant, object-oriented Python architecture.
9. Produce professional documentation suitable for GitHub publication and academic evaluation.

---

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.12+** | Core application language |
| **OpenCV** | Webcam capture, image processing, real-time UI rendering |
| **MediaPipe Face Mesh** | Default landmark-detection backend; requires no external model file |
| **dlib** | Optional classic HOG detector + 68-point `shape_predictor`, matching the original EAR research pipeline |
| **imutils** | `face_utils.shape_to_np` and canonical named landmark index ranges for the dlib backend |
| **NumPy** | Array and coordinate operations across both backends |
| **SciPy** (`scipy.spatial.distance`) | Euclidean distance computation for the EAR formula |
| **playsound** | Lightweight `.wav` alarm playback |
| **Git/GitHub** | Version control and project hosting |

---

## 5. System Architecture

The project follows a **layered, single-responsibility architecture**, deliberately designed so that detection-backend choice, EAR mathematics, and drowsiness-classification logic are all independently swappable and testable:

```
┌───────────────────────────────────────────────────────────────┐
│                           main.py                                │
│  DrowsinessDetectionApp: camera lifecycle, main loop, UI          │
│  orchestration, keyboard input, error handling                    │
└───────┬───────────────────┬───────────────────┬─────────────────┘
        │                   │                   │
┌───────▼────────┐  ┌───────▼─────────┐ ┌───────▼─────────┐
│face_detector.py │  │  eye_tracker.py  │ │drowsiness_       │
│                  │  │                   │ │detector.py        │
│ BaseFaceDetector │  │ EyeAspectRatio-   │ │ Time-based state   │
│  ├─MediaPipe     │  │ Calculator (SciPy)│ │ machine: AWAKE /    │
│  │  FaceDetector  │  │ EARSmoother       │ │ DROWSY, blink        │
│  └─Dlib           │  │ (moving average)  │ │ discrimination        │
│     FaceDetector  │  └───────────────────┘ └───────────────────────┘
│ → FaceLandmarks    │
│  (backend-agnostic) │
└──────────────────────┘
        │
┌───────▼────────┐          ┌──────────────────────┐
│    alarm.py      │          │      fps.py            │
│ Threaded, cooldown│          │  Smoothed FPS counter   │
│ -protected siren   │          └─────────────────────────┘
│ playback             │
└──────────────────────┘
        │
┌───────▼────────┐          ┌──────────────────────┐
│   utils.py       │◄─────────┤    config/settings.py │
│ Drawing helpers,  │          │  Centralized constants│
│ eye-contour        │          │  (sensitivity, camera,│
│ visualization       │          │   colors, UI layout)   │
└──────────────────────┘          └────────────────────────┘
```

**Design principles applied:**

- **Single Responsibility Principle** — `face_detector.py` only detects and standardizes landmark output; `eye_tracker.py` only computes and smooths EAR values; `drowsiness_detector.py` only classifies drowsiness over time; `alarm.py` only manages audio playback; `utils.py` only draws.
- **Backend abstraction** — both `MediaPipeFaceDetector` and `DlibFaceDetector` implement the same `BaseFaceDetector.detect()` interface and return an identical `FaceLandmarks` dataclass, so every downstream module is completely unaware of which backend supplied the data.
- **Zero cross-contamination between logic and rendering** — `eye_tracker.py` and `drowsiness_detector.py` have no dependency on OpenCV drawing calls, MediaPipe, or dlib internals, making them independently unit-testable with synthetic coordinate data and injected timestamps.
- **Configuration over hard-coding** — all magic numbers (EAR threshold, duration thresholds, colors, dimensions, alarm cooldown) live in `config/settings.py`.
- **Typed data contracts** — `FaceLandmarks` and `DrowsinessState` are `dataclasses` providing clean, typed interfaces between pipeline stages.
- **Resource safety and graceful degradation** — the dlib backend auto-downloads its model file if missing (or raises a clear, actionable error), and transparently falls back to MediaPipe if configured to do so; alarm-playback failures are caught and logged without ever crashing the detection loop.

---

## 6. Workflow

The application follows a continuous per-frame processing pipeline:

1. **Capture** — Read a frame from the webcam via OpenCV's `VideoCapture`.
2. **Preprocess** — Horizontally flip the frame for a natural mirror-view experience.
3. **Detect** — Run the configured backend (`MediaPipeFaceDetector` or `DlibFaceDetector`) to obtain 6-point eye-contour landmarks for both eyes, a face bounding box, and a confidence score.
4. **Measure** — Compute the average Eye Aspect Ratio across both eyes via `EyeAspectRatioCalculator`, using `scipy.spatial.distance.euclidean` for each of the three distances in the EAR formula.
5. **Smooth** — Pass the raw EAR through `EARSmoother`, a fixed-window moving average, before it ever reaches the threshold comparison.
6. **Classify** — Feed the smoothed EAR into `DrowsinessDetector.update()`, which tracks how long the eyes have been continuously below the EAR threshold using wall-clock timestamps, classifying short closures as blinks and sustained closures (beyond `DROWSY_DURATION_THRESHOLD_SECONDS`) as drowsiness.
7. **Alert** — If the status is `DROWSY`, call `AlarmManager.trigger()` (which starts a background thread if not already playing and not within its cooldown window) and render a pulsing full-screen visual warning.
8. **Render** — Draw the top status bar (FPS, backend name, webcam status), the live status panel (AWAKE/DROWSY, EAR value, blink count, drowsy-progress bar), the tracking-confidence panel, eye-contour overlays, and the footer.
9. **Loop** — Repeat until the user exits via `Q`, `ESC`, or closes the window.

---

## 7. Implementation

### 7.1 Dual-Backend Face Detection (`face_detector.py`)

`MediaPipeFaceDetector` wraps MediaPipe's Face Mesh solution (refined landmarks, single face), selecting six landmark indices per eye chosen to approximate dlib's classic P1-P6 eye-contour ordering, so the exact same EAR formula applies unchanged regardless of backend. Since MediaPipe's Python API does not expose a true per-frame detection probability for Face Mesh, this backend reports an honestly-labeled "tracking stability" confidence — an exponential moving average of detection success over recent frames — rather than falsely presenting it as a raw model confidence score.

`DlibFaceDetector` wraps dlib's HOG face detector and 68-point `shape_predictor`. Unlike MediaPipe, dlib's `detector.run()` method genuinely exposes a detection score, which is linearly clamped into a 0-1 range for display. Eye landmarks are extracted using `imutils.face_utils.shape_to_np()` and the canonical `FACIAL_LANDMARKS_68_IDXS` named index ranges, avoiding hard-coded "magic" slice bounds.

A `create_face_detector()` factory function selects the configured backend and transparently falls back to MediaPipe if dlib's model file is unavailable and fallback is permitted in settings — logging a clear warning rather than crashing.

### 7.2 Eye Aspect Ratio Computation (`eye_tracker.py`)

`EyeAspectRatioCalculator.compute_ear()` implements the formula exactly:

```
EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
```

using `scipy.spatial.distance.euclidean` for each term. `EARSmoother` applies a fixed-size moving average over recent readings to suppress single-frame landmark jitter before the value is ever compared against a threshold.

### 7.3 Drowsiness State Machine (`drowsiness_detector.py`)

`DrowsinessDetector` tracks the wall-clock timestamp at which the eyes first dropped below the EAR threshold. On each update, if the eyes are still closed, it computes the elapsed closed-duration; once this exceeds `DROWSY_DURATION_THRESHOLD_SECONDS`, the status escalates to `DROWSY`. When the eyes reopen, a closure shorter than `BLINK_MAX_DURATION_SECONDS` is tallied as a normal blink; a longer closure (i.e., a resolved drowsy episode) is correctly excluded from the blink count. A separate `reset()` method clears the in-progress closure timer without affecting the blink tally, specifically for handling temporary tracking dropouts (face briefly out of frame) without misclassifying them as eye closures.

### 7.4 Alarm Management (`alarm.py`)

`AlarmManager.trigger()` starts alarm playback on a daemon thread, guarded by a lock so overlapping triggers can never spawn simultaneous sounds, and enforces a configurable cooldown period so a continuously-drowsy driver isn't spammed with overlapping alarm restarts. All playback exceptions (missing file, unsupported format, missing audio device) are caught inside the thread target and logged as warnings — an alarm failure never propagates to or crashes the main detection loop.

### 7.5 UI Rendering (`utils.py` + `main.py`)

Custom drawing helpers provide translucent rounded panels, drop-shadow text, confidence bars, and eye-contour polygon visualization. `main.py` assembles these into a top status bar, a live status panel (color-coded AWAKE/DROWSY), a tracking-confidence panel, and a pulsing full-screen "DROWSINESS ALERT" banner that activates only while the status remains `DROWSY`.

### 7.6 Error Handling

- Camera initialization retries a bounded number of times before raising a descriptive `CameraNotAvailableError`, caught at the top level in `main()` and reported via `stderr` with a non-zero exit code.
- Per-frame read failures set a "WEBCAM: LOST" UI indicator and trigger a retry loop rather than crashing.
- Missing dlib model files trigger either an automatic download attempt or a clear, actionable `DlibModelUnavailableError` with exact manual-download instructions.
- A top-level `try/except/finally` in `DrowsinessDetectionApp.run()` guarantees resource cleanup (camera release, window destruction, detector closure) even on unexpected exceptions or `Ctrl+C` interruption.

---

## 8. Results

The system reliably computes a stable EAR signal and correctly classifies sustained eye closure as drowsiness while continuing to treat normal blinking as a harmless, separately-tallied event, under typical indoor lighting conditions and real-time frame rates on standard consumer webcams. Deterministic unit testing (using injected timestamps rather than real-time delays) confirmed that: short closures are correctly tallied as blinks without triggering the alarm; sustained closures correctly escalate to `DROWSY` only after crossing the configured duration threshold; a resolved drowsy episode is correctly excluded from the blink counter; and temporary tracking loss is handled via `reset()` without corrupting the blink tally. The alarm system was verified to run entirely off the main thread, respect its cooldown window, and recover cleanly from simulated playback failures without crashing.

---

## 9. Advantages

- Dual-backend architecture provides both immediate out-of-the-box usability (MediaPipe) and academic-research fidelity (dlib), without forcing every user to download a large model file.
- Blink-aware, time-based classification is meaningfully more accurate and realistic than naive consecutive-frame-count tutorials.
- FPS-independent timing ensures consistent behavior across different hardware capabilities.
- Non-blocking, cooldown-protected alarm playback keeps the video feed smooth and responsive even during an active alert.
- Comprehensive, layered error handling means camera issues, missing models, and audio failures degrade gracefully rather than crashing the application.
- Fully adjustable sensitivity via a single settings file, plus a runtime API for dynamic threshold changes.
- Clean, modular, extensible codebase suitable for further research or product development.

---

## 10. Limitations

- Relies on 2D RGB input; extreme head poses, heavy occlusion (e.g., sunglasses), or very poor lighting can degrade landmark accuracy on both backends.
- The MediaPipe backend's "confidence" figure is a tracking-stability heuristic rather than a true per-frame detection probability, as clearly documented in code and README.
- EAR thresholds have some natural variation across individuals due to differing eye shapes and camera angles; the current system uses a single global threshold rather than a per-user calibrated baseline.
- The dlib backend requires a large (~100 MB) external model download not included in the repository, in line with standard practice for this type of asset.
- Detects only eye-closure-based drowsiness; it does not currently incorporate complementary signals such as head-nodding or yawning.

---

## 11. Future Scope

- Add a brief startup calibration phase that learns each user's personal baseline "eyes open" EAR for a more precisely tuned threshold.
- Incorporate head-pose estimation (nod detection) and mouth-aspect-ratio-based yawn detection as additional, complementary drowsiness signals.
- Log all drowsiness events with timestamps to a CSV/database for post-session analytics and reporting.
- Port the alarm system to a richer, cross-platform audio library supporting looping sirens and volume control.
- Explore deployment on embedded hardware (e.g., Raspberry Pi with a dash-mounted camera) for real in-vehicle use.
- Add an automated `pytest` suite with continuous integration covering `eye_tracker.py` and `drowsiness_detector.py` for regression protection.

---

## 12. Conclusion

The AI Drowsiness Detection System successfully demonstrates the integration of research-backed computer vision methodology (the Eye Aspect Ratio technique) with a carefully engineered, dual-backend detection architecture, a time-based signal-processing state machine, and a fully non-blocking alarm system — all assembled with professional software engineering practices. Beyond its immediate function as a safety tool, the project showcases core competencies expected of an applied AI/ML engineer: architectural flexibility between multiple detection technologies, robust handling of real-world sensor noise and failure modes, careful distinction between similar-but-distinct signals (blinking vs. drowsiness), and thorough technical documentation — making it a strong, representative artifact for an AI & Machine Learning Diploma portfolio.
