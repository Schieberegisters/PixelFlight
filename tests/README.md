# Test Suite (`tests/`)

This folder contains the **unit and integration-style tests** for PixelFlight. The suite is organized by domain (`Static`, `Dynamic`, `DroneControl`) and is designed to run reliably on **Windows CI** without requiring real hardware (webcams, drones) by using extensive mocking.

For the global project introduction, environment setup, and dependency versions, see the root `README.md`.

---

## 1) Project Vision & Stack

**Vision:** keep the gesture-recognition pipelines and drone control logic safe to refactor by validating:

- deterministic math (feature extraction, normalization, augmentation)
- dataset loading / split rules (anti-leakage constraints)
- model wiring (custom Keras layers, safe loading)
- control logic mapping (commands, cooldowns, thresholds)

**Stack & CI:** defined at project level. See the root `README.md` and `.github/workflows/ci.yaml`.

---

## 2) Repository Roadmap (The Map)

The test suite is split by feature area:

- `tests/Static/`
  - Validates static gesture utilities, MediaPipe hand landmarker wrappers, the data collector flow, and RandomForest training/persistence.
- `tests/Dynamic/`
  - Validates dynamic gesture dataset collection/augmentation logic, preprocessing split strategy, model creation/loading (ST‑GCN custom layers), and training/evaluation helpers.
- `tests/DroneControl/`
  - Validates DroneControl logic end-to-end using mocks for:
    - model loading (`joblib`, dynamic model loader)
    - MediaPipe annotation/keypoint extraction
    - drone SDK (`djitellopy.Tello`)

Supporting entrypoints:

- `runTests.py` (repo root): convenience wrapper that runs `pytest -v tests ...` and makes sure the project root is on `sys.path`.

---

## 3) Local Setup (The 10-Minute Start)

### Environment setup & running

Use the root `README.md` for venv creation, dependency installation, and how to run the full suite.

---

## 4) Development Workflow

### Branching

Suggested naming:

- `feature/tests-<topic>`
- `fix/tests-<topic>`
- `chore/tests-<topic>`

### Testing style & conventions

- Prefer **pure logic tests** (math, IO orchestration) and use mocks for anything that touches:
  - webcams / OpenCV UI (`cv2.VideoCapture`, `cv2.imshow`, `cv2.waitKey`)
  - MediaPipe inference
  - filesystem-heavy operations (dataset creation, model persistence)
  - drone hardware/SDK
- Use small synthetic arrays and deterministic seeds when relevant.
- Keep tests fast: CI expects this to run as a standard job on `windows-latest`.

### Running a subset

| Goal | Command |
|---|---|
| Only Static tests | `pytest -v tests/Static -p no:cacheprovider --disable-warnings` |
| Only Dynamic tests | `pytest -v tests/Dynamic -p no:cacheprovider --disable-warnings` |
| Only DroneControl tests | `pytest -v tests/DroneControl -p no:cacheprovider --disable-warnings` |

---

## 5) Architectural Patterns

### “No hardware required” principle

Most tests follow a pattern of:

1. Build synthetic inputs (`numpy` arrays, fake landmarks)
2. Patch external dependencies (OpenCV, MediaPipe, model loading)
3. Assert behavior (calls, state transitions, outputs)

This keeps the suite runnable on CI machines with no webcam/drone access.

### Configuration injection via monkeypatch

Some tests (notably `tests/DroneControl/test_drone_control.py`) use `monkeypatch` to override runtime constants (thresholds, sequence lengths, gesture label maps) so tests stay deterministic and aligned with the production logic.

---

## 6) Definition of Done (DoD)

Before opening a PR that changes code covered by `tests/`:

- **Run tests locally**: `python runTests.py`
- **Keep tests deterministic**: no real network/hardware calls, no UI windows required.
- **Update/extend tests** when you change:
  - gesture label sets (`config/gestures.py`)
  - dataset directory layouts or preprocessing rules
  - model loading / custom layer registration
  - DroneControl command mapping logic
- **CI compatibility**: tests must pass on Windows with Python 3.10.x.

---

## Troubleshooting (common)

- **TensorFlow import errors on Windows**: verify you’re using a Python version compatible with `tensorflow==2.15.0` and reinstall dependencies in a clean venv.
- **Hanging tests / OpenCV windows**: ensure any UI calls are patched in tests and you are not running interactive collectors from the test process.
- **Path issues**: prefer running tests from repo root; `runTests.py` already adds the root directory to `sys.path`.

