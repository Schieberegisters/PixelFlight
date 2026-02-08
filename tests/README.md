# Test Suite (`tests/`)

This folder contains the **unit and integration-style tests** for PixelFlight. The suite is organized by domain (`Static`, `Dynamic`, `DroneControl`) and is designed to run reliably on **Windows CI** without requiring real hardware (webcams, drones) by using extensive mocking.

---

## 1) Purpose & Stack

**Vision:** keep the gesture-recognition pipelines and drone control logic safe to refactor by validating:

- deterministic math (feature extraction, normalization, augmentation)
- dataset loading / split rules (anti-leakage constraints)
- model wiring (custom Keras layers, safe loading)
- control logic mapping (commands, cooldowns, thresholds)

**Tooling used in this folder:**

- `pytest` + `unittest.mock`
- synthetic data via `numpy`/`pandas`
- model-related checks via `tensorflow` where needed

---

## 2) Structure

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

### Run commands

| Goal | Command |
|---|---|
| Run all tests (wrapper) | `python runTests.py` |
| Run all tests (pytest) | `pytest -v tests -p no:cacheprovider --disable-warnings` |

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

## Troubleshooting (common)

- **TensorFlow import errors on Windows**: verify you’re using a Python version compatible with `tensorflow==2.15.0` and reinstall dependencies in a clean venv.
- **Hanging tests / OpenCV windows**: ensure any UI calls are patched in tests and you are not running interactive collectors from the test process.
- **Path issues**: prefer running tests from repo root; `runTests.py` already adds the root directory to `sys.path`.

