from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = BACKEND_ROOT / "models" / "asl_wlasl300_realtime.pt"
LABEL_MAP_PATH = BACKEND_ROOT / "artifacts" / "label_map_300.json"
INPUT_DIM = 180
SEQUENCE_LENGTH = 30
MODEL_SOURCE = "asl_wlasl300_realtime"
