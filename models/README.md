# models/ Directory

This directory is where the optional **dlib 68-point facial landmark
model** (`shape_predictor_68_face_landmarks.dat`) belongs, if you
choose to enable the dlib detection backend.

## Why isn't the model file included in the repository?

The file is roughly **~100 MB** uncompressed, is distributed directly
by the dlib project rather than via pip, and is not something that
belongs in version control. This is standard practice for any project
that depends on it.

## You do NOT need this file to run the project

By default, `config/settings.py` sets:

```python
DETECTION_BACKEND = "mediapipe"
```

The MediaPipe backend requires no external model download at all --
it works immediately after `pip install -r requirements.txt`. This
directory and the dlib backend are entirely **optional**.

## Enabling the dlib backend

If you'd like to use the classic dlib HOG + 68-point landmark
pipeline instead (the approach used in the original academic EAR
research), you have two options:

### Option A -- Automatic download (recommended)

Simply set the following in `config/settings.py`:

```python
DETECTION_BACKEND = "dlib"
AUTO_DOWNLOAD_DLIB_MODEL = True   # already the default
```

Then run `python main.py`. On first launch, `src/face_detector.py`
will automatically download and decompress the model file into this
directory for you (requires an internet connection).

### Option B -- Manual download

1. Download the compressed model from the official dlib source:
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
2. Decompress the `.bz2` archive (e.g. `bzip2 -d shape_predictor_68_face_landmarks.dat.bz2`
   on Linux/macOS, or use 7-Zip on Windows).
3. Place the resulting `shape_predictor_68_face_landmarks.dat` file
   directly inside this `models/` directory.
4. Set `DETECTION_BACKEND = "dlib"` in `config/settings.py`.

If the model file is missing and `FALLBACK_TO_MEDIAPIPE_IF_DLIB_UNAVAILABLE`
is `True` (the default) in `config/settings.py`, the application will
automatically and transparently fall back to the MediaPipe backend
with a clear console warning, rather than crashing.
