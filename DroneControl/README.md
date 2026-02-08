# DroneControl (Gesture → Command → Drone)

This folder contains the **runtime controller** that connects gesture recognition to drone movement. It combines:

- **Static gestures** (single-frame hand pose) for mode-like commands (e.g. `HALT`)
- **Dynamic actions** (sequence-based) for movement and special actions (e.g. `FLY_LEFT`, `TAKE_ONOFF`, `FLIP`)

The implementation is designed to run in **debug mode** (no drone connection) or in **live mode** (connects to a DJI Tello via `djitellopy`).

---

## 1) Project Vision & Stack

**Vision:** translate human hand/body gestures into safe, stable drone commands in real time:

- keep control latency low (webcam → inference → command mapping)
- avoid command flicker with stability logic for dynamic actions
- prevent repeated dangerous actions via cooldown + threading

**Core stack (see repo `requirements.txt`):**

- **Python**: (recommended) 3.10+
- **OpenCV**: `opencv-python` (camera input + UI)
- **MediaPipe**: `mediapipe==0.10.9` (Holistic pose/hands)
- **TensorFlow/Keras**: `tensorflow==2.15.0` (dynamic model inference)
- **scikit-learn + joblib**: static model inference (`predict_proba`)
- **djitellopy**: DJI Tello SDK client

---

## 2) Repository Roadmap (The Map)

Inside `DroneControl/`:

- `drone_control.py`
  - Main class `DroneControl`
  - Opens webcam, runs MediaPipe Holistic, executes static + dynamic prediction
  - Maps gestures/actions to `CurrentCommand` and dispatches Tello commands
  - Supports `debug=True` mode (no drone connection)
- `gestureRecognition.py`
  - `landmark_normalization(...)`: produces a **15-feature** vector from 21 hand landmarks (distance-based)
- `States.py`
  - `CurrentCommand`: enum used as a shared command state machine
- `__init__.py`
  - package exports

Key dependencies outside this folder:

- `config/drone.py`: thresholds, cooldown, velocity scaling factors, model artifact paths
- `config/gestures.py`: label maps (`STATIC_HAND_GESTURES`, `DYNAMIC_ACTIONS`)
- `config/dynamic.py`: `SEQUENCE_LENGTH`, `STABLE_LENGTH` used to stabilize dynamic actions
- `Dynamic/` module:
  - `Dynamic.mediapipe_utils.annotate_frame` + `mp_holistic`
  - `Dynamic.keypoints.extract_keypoints` (1662-dim vector)
  - `Dynamic.model.load_model` (dynamic action model)

---

## 3) Local Setup (The 10-Minute Start)

### Prerequisites

- A webcam (OpenCV uses camera index `0` by default)
- (Live mode) a DJI Tello on the same network and `djitellopy` working

### Setup commands (Windows / PowerShell)

| Goal | Command |
|---|---|
| Create venv | `python -m venv .venv` |
| Activate venv | `.\.venv\Scripts\Activate.ps1` |
| Install deps | `pip install -r requirements.txt` |

### Run in debug mode (recommended first)

Debug mode shows the UI windows but **does not connect to the drone**:

| Goal | Command |
|---|---|
| Start controller in debug | `python main.py --debug` |

Quit with `q`.

### Run in live mode

Live mode connects to the Tello and sends RC commands / special actions:

| Goal | Command |
|---|---|
| Start controller (live) | `python main.py` |

---

## 4) Development Workflow

### Branching

Suggested naming:

- `feature/dronecontrol-<topic>`
- `fix/dronecontrol-<topic>`
- `chore/dronecontrol-<topic>`

### Coding standards (pragmatic)

- Keep runtime constants in `config/` (thresholds, velocities, model paths).
- Prefer **pure mapping functions** and small methods for testability.
- Any change to gesture/action labels must stay consistent with `config/gestures.py`.

### Running tests

| Goal | Command |
|---|---|
| Run all tests | `python runTests.py` |
| Run DroneControl tests only | `pytest -v tests/DroneControl -p no:cacheprovider --disable-warnings` |

---

## 5) Architectural Patterns

### End-to-end runtime flow

```mermaid
flowchart LR
  A[Webcam frame] --> B[MediaPipe Holistic via Dynamic.annotate_frame]
  B --> C[Static: right-hand landmarks -> 15 features -> joblib model]
  B --> D[Dynamic: keypoints 1662/frame -> deque window -> TF model]
  C --> E[static_gesture]
  D --> F[dynamic_action (stability-gated)]
  E --> G[_handle_gesture_logic]
  F --> G
  G --> H[Command state: CurrentCommand]
  H --> I[Dispatch: RC control or special thread + cooldown]
```

### Stability gating for dynamic actions

Dynamic predictions are only accepted when:

- the controller has collected a full window of length `SEQUENCE_LENGTH`
- the predicted class index stays constant for `STABLE_LENGTH` consecutive predictions

This avoids jitter from frame-to-frame noise.

### Safety & special actions

Special actions (`TAKE_ONOFF`, `FLIP`) are executed in a **background thread** and throttled by a **cooldown** (`COOLDOWN_SECONDS`) to avoid repeated takeoff/land/flip triggers.

### Command mapping rules (high level)

- If static gesture is `HALT`, movement is immediately suppressed (commands reset to IDLE for that cycle).
- Movement actions map to directional commands (`FLY_LEFT`, `FLY_UP`, …).
- Motor power is derived from `VELOCITY_FACTOR_X/Y/Z` and the current command state.

---

## 6) Definition of Done (DoD)

Before opening a PR that touches `DroneControl/`:

- **Run tests**: `pytest -v tests/DroneControl ...`
- **Debug run works**: `python main.py --debug` starts and exits cleanly with `q`
- **No hardware dependency in tests**: Drone, models, OpenCV, and MediaPipe must be mockable
- **Config consistency**: label maps and sequence lengths remain aligned with:
  - `config/gestures.py`
  - `config/dynamic.py`
  - `config/drone.py`
- **Safety preserved**: cooldown/threading for special actions is not bypassed

---

## Notes on model paths

Default artifact paths come from `config/drone.py`:

- Static model: `Static/models/140k.joblib`
- Dynamic model: `Dynamic/models/actionNoFlip.keras`

If you move/rename artifacts, update `config/drone.py` accordingly.

