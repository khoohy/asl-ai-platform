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
TOP_K = 5
HAND_LOSS_GRACE_FRAMES = 10
IDLE_STATUS = "idle"
HOLDING_CONTEXT_STATUS = "holding_context"
WAITING_FOR_HANDS_STATUS = "waiting_for_hands"
TRANSITIONING_STATUS = "transitioning"
COLLECTING_EVIDENCE_STATUS = "collecting_evidence"
BASE_CONFIDENCE_THRESHOLD = 0.65
ADAPTIVE_CONFIDENCE_FLOOR = 0.45
RUNNER_UP_MARGIN = 0.12
STABILIZATION_WINDOW = 10
MIN_VOTE_COUNT = 6
STABLE_OUTPUT_HOLD_FRAMES = 8
MIN_FRAMES_BETWEEN_STABLE_OUTPUTS = 4
TRANSITION_COOLDOWN_FRAMES = 6
PEAK_HISTORY_WINDOW = 5
PEAK_MIN_COUNT = 2
PEAK_MARGIN = 0.10

SIGN_CONFIDENCE_OVERRIDES = {
    "APPROVE": 0.58,
    "ARRIVE": 0.58,
    "BAD": 0.58,
    "BEFORE": 0.56,
    "BOOK": 0.57,
    "CATCH": 0.52,
    "CENTER": 0.58,
    "CHANGE": 0.55,
    "CHRISTMAS": 0.55,
    "COLLEGE": 0.57,
    "COOK": 0.56,
    "COPY": 0.56,
    "EASY": 0.56,
    "FINE": 0.56,
    "HOPE": 0.56,
    "JACKET": 0.60,
    "LAW": 0.57,
    "MILK": 0.56,
    "MUSIC": 0.56,
    "ORDER": 0.56,
    "RABBIT": 0.54,
    "SIGN": 0.54,
}

PEAK_SIGN_CONFIDENCE_OVERRIDES = {
    "ARRIVE": 0.72,
    "CATCH": 0.68,
    "CHRISTMAS": 0.62,
    "HOPE": 0.72,
    "JACKET": 0.80,
    "LAW": 0.72,
}

MOTION_REQUIRED_SIGNS = {
    "AGAIN": 0.016,
    "ARRIVE": 0.016,
    "CATCH": 0.016,
    "CEREAL": 0.016,
    "CRASH": 0.018,
    "DOCTOR": 0.018,
    "EAT": 0.016,
    "HEARING": 0.016,
    "HOUR": 0.015,
    "LAW": 0.016,
    "TEST": 0.016,
}

CONFUSION_PAIRS = {
    "AGAIN": {"DOCTOR"},
    "APPROVE": {"DRAW"},
    "ARRIVE": {"BABY"},
    "BAD": {"SCHOOL"},
    "BAR": {"BEHIND"},
    "BEFORE": {"CLOSE", "WINDOW"},
    "BEHIND": {"BAR", "WITH"},
    "CATCH": {"YEAR"},
    "CENTER": {"DOCTOR"},
    "CHEESE": {"SCHOOL"},
    "CHILDREN": {"SHORT"},
    "CORN": {"SIGN", "CHAIR"},
    "CRASH": {"ACCIDENT", "PROBLEM"},
    "CRAZY": {"COUSIN"},
    "DARK": {"MATCH"},
    "BUT": {"DIFFERENT", "TEST"},
    "DOCTOR": {"AGAIN", "LEARN", "CENTER"},
    "EXAMPLE": {"SHOW"},
    "FAR": {"GAME"},
    "FEEL": {"HAPPY", "HEART"},
    "FINE": {"TABLE", "HAPPY", "HEART"},
    "FINISH": {"DRESS", "CLOTHES"},
    "FIRST": {"YEAR", "PROBLEM"},
    "HEARING": {"HEAR"},
    "HOPE": {"WAR"},
    "HOUR": {"LAW"},
    "JACKET": {"HOW", "BATH"},
    "KISS": {"MORE"},
    "LAW": {"HOUR"},
    "MUSIC": {"CHAIR", "SIGN"},
    "PENCIL": {"WRITE", "SECRETARY"},
    "RABBIT": {"YEAR"},
    "SAME": {"STAY"},
    "SECRETARY": {"PENCIL"},
    "SIGN": {"CHAMPION"},
    "TEST": {"DIFFERENT", "BUT"},
    "WRITE": {"PENCIL"},
}
