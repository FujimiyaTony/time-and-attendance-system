"""Chong gia mach:
   1) TextureCNN: phan tich ket cau be mat (vien man hinh, phan chieu, moire, vet in...)
   2) Challenge-response: chop mat + xoay dau ngau nhien (MediaPipe FaceMesh + OpenCV)
"""
import random
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import config


# ---------------------------------------------------------------- CNN
class TextureCNN(nn.Module):
    """CNN nho, dau ra 1 logit = xac suat GIA MAO (spoof). Real score = 1 - sigmoid(logit)."""

    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.features(x))


def crop_face(frame_bgr, bbox, scale=config.CROP_SCALE):
    """Cat vuong quanh mat, mo rong `scale` lan de giu lai boi canh (vien man hinh, tay cam may...)."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * scale / 2
    a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
    c, d = int(min(cx + half, w)), int(min(cy + half, h))
    crop = frame_bgr[b:d, a:c]
    return cv2.resize(crop, (config.CNN_INPUT, config.CNN_INPUT)) if crop.size else None


class TextureLiveness:
    def __init__(self):
        self.ok = False
        self.model = TextureCNN()
        if config.LIVENESS_MODEL.exists():
            try:
                state = torch.load(
                    config.LIVENESS_MODEL, map_location="cpu", weights_only=True
                )
                self.model.load_state_dict(state)
                self.model.eval()
                self.ok = True
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                print(f"[CANH BAO] Model liveness khong hop le: {exc}")
        else:
            print("[CANH BAO] Chua co models/liveness_cnn.pt -> chi dung thu thach chop mat/xoay dau.\n"
                  "           Chay collect_data.py + train_liveness.py de huan luyen CNN.")

    @torch.no_grad()
    def predict(self, crop_bgr) -> float:
        """Xac suat la nguoi THAT (0..1)."""
        if not self.ok or crop_bgr is None:
            return 1.0
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).float().div(255).permute(2, 0, 1).unsqueeze(0)
        return 1.0 - torch.sigmoid(self.model(x)).item()


# ---------------------------------------------------------------- Challenge
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
NOSE, CHEEK_L, CHEEK_R = 1, 234, 454

PROMPTS = {
    "look_front": "Hay NHIN THANG CAMERA",
    "turn_left": "Hay XOAY DAU SANG TRAI",
    "turn_right": "Hay XOAY DAU SANG PHAI",
}


def _ear(pts):
    p = np.array(pts)
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = 2.0 * np.linalg.norm(p[0] - p[3])
    return v / h if h > 0 else 1.0


def eye_aspect_ratio(lm, w, h):
    def pts(ids):
        return [(lm[i].x * w, lm[i].y * h) for i in ids]
    return (_ear(pts(RIGHT_EYE)) + _ear(pts(LEFT_EYE))) / 2


def head_ratio(lm):
    """0.5 = nhin thang; <0.5 mui lech ve ben trai anh; >0.5 ve ben phai (anh da lat guong)."""
    width = lm[CHEEK_R].x - lm[CHEEK_L].x
    return (lm[NOSE].x - lm[CHEEK_L].x) / width if width > 1e-6 else 0.5


class Challenge:
    """Chuoi thu thach ngau nhien, ket thuc bang thao tac nhin thang de chup anh chinh dien."""

    def __init__(self):
        self.steps = [random.choice(["turn_left", "turn_right"]), "look_front"]
        self.idx = 0
        self.t0 = time.time()
        self.closed = 0
        self.hold = 0
        self.tex_scores: list[float] = []

    @property
    def current(self):
        return self.steps[self.idx] if self.idx < len(self.steps) else None

    @property
    def prompt(self):
        return PROMPTS.get(self.current, "")

    @property
    def mean_texture(self):
        return float(np.mean(self.tex_scores)) if self.tex_scores else 0.0

    def _next(self):
        self.idx += 1
        self.closed = self.hold = 0

    def update(self, lm, frame_shape, tex_score):
        """Tra ve 'running' | 'passed' | 'timeout'."""
        self.tex_scores.append(tex_score)
        if time.time() - self.t0 > config.CHALLENGE_TIMEOUT:
            return "timeout"
        if lm is not None and self.current:
            h, w = frame_shape[:2]
            r = head_ratio(lm)
            if self.current == "look_front":
                ok = config.TURN_LEFT_RATIO <= r <= config.TURN_RIGHT_RATIO
            else:
                ok = (r < config.TURN_LEFT_RATIO) if self.current == "turn_left" \
                    else (r > config.TURN_RIGHT_RATIO)
            self.hold = self.hold + 1 if ok else 0
            if self.hold >= config.TURN_HOLD_FRAMES:
                self._next()
        return "passed" if self.current is None else "running"
