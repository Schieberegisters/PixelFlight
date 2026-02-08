# PixelFlight

PixelFlight is a gesture-driven control system that combines **computer vision** and **machine learning** to translate **static hand poses** and **dynamic motion sequences** into drone commands (DJI Tello via `djitellopy`).  
The repository is split into four main areas: `Static/`, `Dynamic/`, `DroneControl/`, and `tests/`.

---

## 1) Project Vision & Stack

**Vision:** control a drone in real time using human gestures, with a pragmatic focus on:

- **low-latency inference** (webcam → keypoints → prediction)
- **stable commands** (temporal smoothing for dynamic actions)
- **safety** (cooldowns + special-action threading, and a `HALT` static override)
- **refactor confidence** (a Windows CI test suite that avoids real hardware)

**Core stack (see `requirements.txt`):**

- **Python**: (recommended) 3.10+ (CI uses 3.10.9)
- **TensorFlow/Keras**: `tensorflow==2.15.0` (dynamic model inference/training)
- **MediaPipe**: `mediapipe==0.10.9`
  - **Holistic** (pose + hands) for dynamic keypoints
  - **Tasks HandLandmarker** for static hand landmark extraction
- **OpenCV**: `opencv-python` (webcam + visualization)
- **scikit-learn + joblib**: static model training/inference
- **djitellopy**: DJI Tello drone control
- **pytest**: test suite (runs on Windows CI)

---

## 2) Repository Roadmap (The Map)

High-level structure and where the “real work” lives:

- **`DroneControl/`**: runtime controller that connects gestures → command mapping → drone dispatch  
  - Entry: `main.py` (`--debug` to run without a drone)  
  - More: `DroneControl/README.md`
- **`Dynamic/`**: sequence-based action recognition pipeline (dataset → preprocessing → training/eval)  
  - Default model: **ST‑GCN** (custom Keras layers)  
  - More: `Dynamic/README.md`
- **`Static/`**: single-frame hand gesture recognition (MediaPipe HandLandmarker + 15-D features + RandomForest)  
  - More: `Static/README.md`
- **`tests/`**: unit + integration-style tests organized by domain  
  - More: `tests/README.md`
- **`config/`**: configuration constants (model paths, thresholds, sequence lengths, label maps)

---

## 3) Local Setup (The 10-Minute Start)

### Prerequisites

- Python + pip
- A webcam (OpenCV uses device index `0` by default)
- (Optional, live control) a DJI Tello on the same network

### Setup commands (Windows / PowerShell)

| Goal | Command |
|---|---|
| Create venv | `python -m venv .venv` |
| Activate venv | `.\.venv\Scripts\Activate.ps1` |
| Install dependencies | `pip install -r requirements.txt` |

### Run the controller (debug first)

| Goal | Command |
|---|---|
| Start in debug mode (no drone connection) | `python main.py --debug` |
| Start in live mode (connect to Tello) | `python main.py` |

Quit with `q`.

### Run tests

| Goal | Command |
|---|---|
| Run full suite | `python runTests.py` |
| Run via pytest | `pytest -v tests -p no:cacheprovider --disable-warnings` |

---

## 4) Development Workflow

### Branching

Suggested naming:

- `feature/<area>-<topic>` (e.g. `feature/dynamic-augmentations`)
- `fix/<area>-<topic>` (e.g. `fix/dronecontrol-cooldown`)
- `chore/<area>-<topic>` (e.g. `chore/tests-cleanup`)

### Coding standards (pragmatic)

- Keep **constants and artifact paths** in `config/` (avoid hardcoding paths in scripts).
- Keep **labels consistent**:
  - static labels: `config/gestures.py` (`STATIC_HAND_GESTURES`)
  - dynamic labels: `config/gestures.py` (`DYNAMIC_ACTIONS`)
- Prefer small, testable functions; tests frequently mock OpenCV/MediaPipe and model loading.

### Testing strategy

- Run tests whenever you touch:
  - feature extraction / preprocessing logic
  - model loading or custom layers
  - DroneControl command mapping / timing

See `tests/README.md` for patterns and troubleshooting.

---

## 5) Architectural Patterns

### System overview

```mermaid
flowchart LR
  A[Webcam frame] --> B[MediaPipe]
  B --> C1[Static pipeline\nHandLandmarker -> 15-D features -> joblib model]
  B --> C2[Dynamic pipeline\nHolistic -> 1662 keypoints/frame -> TF model]
  C1 --> D[static_gesture]
  C2 --> E[dynamic_action\n(stability gated)]
  D --> F[DroneControl mapping]
  E --> F
  F --> G[Dispatch\nRC control or special thread + cooldown]
  G --> H[Tello drone\n(live mode)]
```

### Static vs Dynamic responsibilities

- **Static** (`Static/`): classifies a *single* hand pose into a discrete label (fast, lightweight).
- **Dynamic** (`Dynamic/`): classifies *motion over time* (sequence window + stability filter).
- **DroneControl** (`DroneControl/`): merges both signals and enforces safety rules (cooldown, `HALT` override).

### Dynamic stability gating (why it matters)

Dynamic actions are accepted only after:

- filling a sequence buffer of length `SEQUENCE_LENGTH`
- holding the same predicted class for `STABLE_LENGTH` consecutive predictions

This reduces command flicker from noisy frames.

---

## 6) Definition of Done (DoD)

Before opening a Pull Request:

- **Tests pass**: `python runTests.py`
- **Docs updated** when changing behavior:
  - `Static/README.md` for feature vectors / CSV schema / assets
  - `Dynamic/README.md` for dataset layout / split policy / model entrypoint
  - `DroneControl/README.md` for runtime mapping, safety, and modes
  - `tests/README.md` for suite structure or conventions
- **No silent label drift**: keep label names and indices aligned with `config/gestures.py`
- **Safety preserved**: don’t bypass cooldown/threading for special actions in live mode

---

## Module docs (recommended reading)

- `DroneControl/README.md` — how to run debug/live and how mapping + cooldown works
- `Dynamic/README.md` — dataset layout, training/evaluation, ST‑GCN notes
- `Static/README.md` — HandLandmarker usage, 15‑D feature vector, RandomForest training
- `tests/README.md` — how to run and extend tests (Windows CI-friendly)
