from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

FAISS_INDEX = DATA_DIR / "faces.index"
NAMES_FILE = DATA_DIR / "names.json"
DB_PATH = DATA_DIR / "attendance.db"
ATTENDANCE_PHOTO_DIR = DATA_DIR / "attendance_photos"
LIVENESS_MODEL = BASE_DIR / "models" / "liveness_cnn.pt"
LIVENESS_DATASET = BASE_DIR / "dataset"          # dataset/real, dataset/spoof

# Camera
CAMERA_ID = 0
FRAME_W, FRAME_H = 640, 480

# Face recognition (ArcFace qua InsightFace)
INSIGHTFACE_MODEL = "buffalo_l"      # ArcFace R50; dung "buffalo_s" neu may yeu
DET_SIZE = (320, 320)
EMB_DIM = 512
RECOG_THRESHOLD = 0.40               # cosine similarity toi thieu (0.35-0.5 tuy du lieu)
MIN_FACE_SIZE = 110                  # pixel, mat qua nho thi bao lai gan hon

# Liveness
LIVENESS_THRESHOLD = 0.60            # diem "that" trung binh cua CNN
CNN_INPUT = 80
CROP_SCALE = 2.0                     # cat rong hon mat de CNN thay vien man hinh / vien giay
EAR_THRESH = 0.21                    # Eye Aspect Ratio, duoi muc nay = nham mat
EAR_CONSEC_FRAMES = 2
TURN_LEFT_RATIO = 0.38
TURN_RIGHT_RATIO = 0.62
TURN_HOLD_FRAMES = 3
CHALLENGE_TIMEOUT = 18.0             # giay de hoan thanh hai thao tac ngau nhien

# Cham cong
DEPARTMENTS = ("BG6", "BU7", "BU10")
COOLDOWN_SECONDS = 60                # khong ghi trung trong khoang nay
RESULT_DISPLAY_SECONDS = 3.0
