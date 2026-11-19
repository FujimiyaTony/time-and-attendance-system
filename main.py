"""He thong cham cong nhan dien khuon mat + chong gia mach.
   python main.py          # chay cham cong
   python main.py export   # xuat attendance.xlsx kem anh
Phim Q de thoat.
"""
import sys
import time
from collections import Counter
import cv2
import mediapipe as mp
import numpy as np
import config
import attendance
from face_engine import FaceEngine
from face_db import FaceDB
from liveness import TextureLiveness, Challenge, crop_face

GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 220), (0, 220, 255)


def put(img, text, y, color=YELLOW, scale=0.7):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def identify(db, embs):
    """Nhan dien tren nhieu frame, yeu cau da so frame cung 1 nguoi (chong tron/doi mat giua chung)."""
    pick = [embs[i] for i in np.linspace(0, len(embs) - 1, min(len(embs), 15)).astype(int)]
    results = [db.search(e) for e in pick]
    names = [n for n, _ in results if n]
    if not names:
        return None, 0.0
    name, cnt = Counter(names).most_common(1)[0]
    if cnt / len(results) < 0.6:
        return None, 0.0
    return name, float(np.mean([s for n, s in results if n == name]))


def run():
    engine, db, texture = FaceEngine(), FaceDB(), TextureLiveness()
    if db.index.ntotal == 0:
        print("Chua co ai trong CSDL. Chay: python enroll.py --name <ten>")
    mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False,
                                           min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(config.CAMERA_ID)
    if not cap.isOpened():
        print(f"[LOI] Khong mo duoc camera ID {config.CAMERA_ID}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)

    state, challenge, embs = "WAIT", None, []
    result_until, result_lines, result_color = 0, [], GREEN

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        view = frame.copy()
        now = time.time()

        if state == "RESULT":
            if now > result_until:
                state = "WAIT"
            for i, line in enumerate(result_lines):
                put(view, line, 30 + 30 * i, result_color)
            cv2.imshow("Cham cong", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        face = engine.largest_face(frame)
        if face is None:
            state, challenge, embs = "WAIT", None, []
            put(view, "Khong thay khuon mat", 30, RED)
        else:
            x1, y1, x2, y2 = face.bbox.astype(int)
            if min(x2 - x1, y2 - y1) < config.MIN_FACE_SIZE:
                state, challenge, embs = "WAIT", None, []
                put(view, "Hay lai gan camera hon", 30, RED)
                cv2.rectangle(view, (x1, y1), (x2, y2), RED, 2)
            else:
                if state == "WAIT":
                    challenge, embs, state = Challenge(), [], "CHALLENGE"

                res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                lm = res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
                tex = texture.predict(crop_face(frame, face.bbox))
                embs.append(engine.embedding(face))
                status = challenge.update(lm, frame.shape, tex)

                cv2.rectangle(view, (x1, y1), (x2, y2), YELLOW, 2)
                put(view, challenge.prompt, 30)
                put(view, f"Texture: {tex:.2f}  Buoc {challenge.idx + 1}/{len(challenge.steps)}", 60,
                    GREEN if tex >= config.LIVENESS_THRESHOLD else RED, 0.55)

                if status == "timeout":
                    state, result_color = "RESULT", RED
                    result_lines = ["THAT BAI: khong hoan thanh thu thach", "(co the la anh/video)"]
                elif status == "passed":
                    state = "RESULT"
                    if texture.ok and challenge.mean_texture < config.LIVENESS_THRESHOLD:
                        result_color = RED
                        result_lines = ["GIA MAO: ket cau anh bat thuong",
                                        f"(diem that TB = {challenge.mean_texture:.2f})"]
                    else:
                        name, score = identify(db, embs)
                        if name is None:
                            result_color = RED
                            result_lines = ["Khong nhan ra khuon mat", "(chua dang ky?)"]
                        else:
                            try:
                                kind = attendance.log(name, score, frame, face.bbox)
                            except ValueError as exc:
                                result_color = RED
                                result_lines = [str(exc)]
                            else:
                                result_color = GREEN
                                if kind:
                                    result_lines = [f"{name} - {kind}", f"Do khop: {score:.2f}"]
                                    print(f"[{time.strftime('%H:%M:%S')}] {name} {kind} (score={score:.2f})")
                                else:
                                    result_lines = [f"{name}: da cham cong gan day"]
                if state == "RESULT":
                    result_until = now + config.RESULT_DISPLAY_SECONDS
                    challenge, embs = None, []

        cv2.imshow("Cham cong", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        if len(sys.argv) > 3:
            raise SystemExit("Cach dung: python main.py export [BG6|BU7|BU10]")
        department = sys.argv[2] if len(sys.argv) == 3 else None
        try:
            attendance.export_xlsx(department=department)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        run()
