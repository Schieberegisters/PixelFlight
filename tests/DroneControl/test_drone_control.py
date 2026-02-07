import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

from DroneControl.drone_control import DroneControl
from DroneControl.States import CurrentCommand

# --- MOCK PATHS ---
MOCK_STATIC_PATH = "tests/fixtures/mock_static_model.pkl"
MOCK_DYNAMIC_PATH = "tests/fixtures/mock_dynamic_model.keras"
MOCK_LANDMARKER_PATH = "tests/fixtures/mock_landmarker.task"

# --- CONFIGURATION MOCK ---
@pytest.fixture(autouse=True)
def mock_config(monkeypatch):
    """Injects REAL gesture definitions and test thresholds."""
    monkeypatch.setattr("DroneControl.drone_control.PRED_THRESHOLD_STATIC", 0.7)
    monkeypatch.setattr("DroneControl.drone_control.COOLDOWN_SECONDS", 2.0)
    monkeypatch.setattr("DroneControl.drone_control.VELOCITY_FACTOR_X", 0.5)
    monkeypatch.setattr("DroneControl.drone_control.VELOCITY_FACTOR_Y", 0.5)
    monkeypatch.setattr("DroneControl.drone_control.VELOCITY_FACTOR_Z", 0.5)
    monkeypatch.setattr("DroneControl.drone_control.SEQUENCE_LENGTH", 5)
    monkeypatch.setattr("DroneControl.drone_control.STABLE_LENGTH", 3)
    
    # Matching the indices used in the main DroneControl logic
    real_static = {0: "BACKWARDS", 1: "FORWARD", 2: "HALT", 3: "TAKEOFF", 4: "XYCONTROL"}
    real_dynamic = np.array([
        "TAKE_ONOFF", "FLY_LEFT", "FLY_RIGHT", "FLY_UP", "FLY_DOWN",
        "FLY_BACK", "FLY_FORWARD", "ROTATE", "FLIP", "IDLE"
    ])
    
    monkeypatch.setattr("DroneControl.drone_control.STATIC_HAND_GESTURES", real_static)
    monkeypatch.setattr("DroneControl.drone_control.DYNAMIC_ACTIONS", real_dynamic)


class TestDroneControl:

    @pytest.fixture
    def mock_dependencies(self):
        """Mocks hardware, ML Models, and MediaPipe components."""
        with patch("DroneControl.drone_control.joblib.load") as mock_joblib, \
            patch("DroneControl.drone_control.load_dynamic_model") as mock_load_dyn, \
            patch("DroneControl.drone_control.Tello") as mock_tello_cls, \
            patch("DroneControl.drone_control.cv2.VideoCapture") as mock_cap, \
            patch("DroneControl.drone_control.mp_holistic.Holistic") as mock_holistic, \
            patch("DroneControl.drone_control.annotate_frame") as mock_annotate, \
            patch("DroneControl.drone_control.extract_keypoints") as mock_extract, \
            patch("DroneControl.drone_control.landmark_normalization") as mock_norm, \
            patch("mediapipe.solutions.drawing_utils.draw_landmarks") as mock_draw:
            
            # Setup Static Model
            mock_static_model = MagicMock()
            # Default return to a safe gesture (Index 4: XYCONTROL) to avoid HALT overrides in generic tests
            mock_static_model.predict_proba.return_value = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
            mock_joblib.return_value = mock_static_model
            
            # Setup Dynamic Model
            mock_dynamic_model = MagicMock()
            mock_load_dyn.return_value = mock_dynamic_model
            
            # Setup Annotation Result Mock
            mock_results = MagicMock()
            mock_results.pose_landmarks = None
            mock_results.left_hand_landmarks = None
            mock_right_hand = MagicMock()
            mock_right_hand.landmark = [MagicMock(x=0.5, y=0.5, z=0.0) for _ in range(21)]
            mock_results.right_hand_landmarks = mock_right_hand
            
            mock_annotate.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), mock_results)
            
            yield {
                "joblib": mock_joblib,
                "load_dyn": mock_load_dyn,
                "static_model": mock_static_model,
                "dynamic_model": mock_dynamic_model,
                "tello_cls": mock_tello_cls,
                "annotate": mock_annotate,
                "extract": mock_extract,
                "norm": mock_norm,
                "draw": mock_draw,
                "results": mock_results
            }

    def test_initialization_paths(self, mock_dependencies):
        """Verify model loading."""
        dc = DroneControl(
            debug=True,
            static_model_path=MOCK_STATIC_PATH,
            dynamic_model_path=MOCK_DYNAMIC_PATH,
            landmarker_model_path=MOCK_LANDMARKER_PATH
        )
        mock_dependencies["joblib"].assert_called_with(MOCK_STATIC_PATH)
        mock_dependencies["load_dyn"].assert_called_with(MOCK_DYNAMIC_PATH)

    def test_predict_static_forward(self, mock_dependencies):
        """Test FORWARD prediction."""
        dc = DroneControl(debug=True)
        # Force "FORWARD" (Index 1) for this specific test
        mock_dependencies["static_model"].predict_proba.return_value = np.array([[0.05, 0.9, 0.05, 0.0, 0.0]])
        mock_dependencies["norm"].return_value = [0.1] * 63
        
        result = dc._predict_static([MagicMock()])
        assert result == "FORWARD"

    def test_predict_dynamic_fly_left(self, mock_dependencies):
        """Test 'FLY_LEFT' with buffer fill logic."""
        dc = DroneControl(debug=True)
        mock_model = mock_dependencies["dynamic_model"]
        
        prediction_probs = np.zeros((1, 10))
        prediction_probs[0, 1] = 0.95 # Index 1 = FLY_LEFT
        mock_model.predict.return_value = prediction_probs
        
        for _ in range(4):
            assert dc._predict_dynamic([0.0]*10) is None
        
        assert dc._predict_dynamic([0.0]*10) is None
        assert dc._predict_dynamic([0.0]*10) is None
        
        result = dc._predict_dynamic([0.0]*10)
        assert result == "FLY_LEFT"

    def test_handle_gesture_logic_movement(self, mock_dependencies):
        """Test dynamic movement mapping in _handle_gesture_logic."""
        dc = DroneControl(debug=True)
        dc._handle_gesture_logic(static_gesture=None, dynamic_action="FLY_UP")
        assert dc.command_ud == CurrentCommand.FLY_UP

    def test_handle_gesture_logic_special(self, mock_dependencies):
        """Test special actions (TAKE_ONOFF, FLIP) in _handle_gesture_logic."""
        dc = DroneControl(debug=False)
        dc.drone = MagicMock()
        dc.drone.is_flying = False
        
        dc._handle_gesture_logic(static_gesture=None, dynamic_action="TAKE_ONOFF")
        assert dc.takeoff == CurrentCommand.TAKE_OFF
        
        dc._handle_gesture_logic(static_gesture=None, dynamic_action="FLIP")
        assert dc.flip == CurrentCommand.FLIP

    def test_motor_power_calculation(self, mock_dependencies):
        """Test RC integer values based on CurrentCommand."""
        dc = DroneControl(debug=True)
        dc.command_ud = CurrentCommand.FLY_UP
        dc.command_lr = CurrentCommand.FLY_RIGHT
        
        lr, fb, ud, yv = dc._calculate_motor_power()
        # 100 * 0.5 factor
        assert lr == 50
        assert ud == 50

    def test_full_pipeline_dynamic(self, mock_dependencies):
        """Integration test: Frames -> Dynamic Model -> Logic."""
        dc = DroneControl(debug=True)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Setup Dynamic Prediction: Index 2 = FLY_RIGHT
        mock_dyn_probs = np.zeros((1, 10))
        mock_dyn_probs[0, 2] = 0.99 
        mock_dependencies["dynamic_model"].predict.return_value = mock_dyn_probs
        
        # Setup Static Prediction: XYCONTROL (Index 4) 
        # avoid HALT (Index 2) because HALT triggers an early 'return' which ignores dynamic actions
        mock_dependencies["static_model"].predict_proba.return_value = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
        
        mock_dependencies["extract"].return_value = np.zeros(1662)
        mock_dependencies["norm"].return_value = [0.0] * 63

        with patch("cv2.imshow"):
            # Call process_frame 12 times to satisfy SEQUENCE_LENGTH and STABLE_LENGTH
            for _ in range(12):
                dc._process_frame(dummy_frame)
        
        assert dc.command_lr == CurrentCommand.FLY_RIGHT


def run_tests_directly() -> None:
    """Entry point for direct script execution."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()