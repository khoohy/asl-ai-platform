from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = BACKEND_ROOT / "models" / "asl_wlasl300_realtime.pt"
LABEL_MAP_PATH = BACKEND_ROOT / "artifacts" / "label_map_300.json"
INPUT_DIM = 180
SEQUENCE_LENGTH = 30
MODEL_SOURCE = "asl_wlasl300_realtime"
HAND_FEATURE_DIM = 126
POSE_FEATURE_DIM = 21
FACE_FEATURE_DIM = 33
SELECTED_POSE_LANDMARKS = [0, 11, 12, 13, 14, 15, 16]
SELECTED_FACE_LANDMARKS = [10, 151, 168, 1, 2, 13, 14, 17, 152, 33, 263]
