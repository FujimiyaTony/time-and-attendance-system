"""Dang ky khuon mat.
   python enroll.py --name "Tony" --part BG6       # chup tu webcam (SPACE chup, Q thoat)
   python enroll.py --name "Tony" --part BU7 --images anh/
"""
import argparse
import time
from pathlib import Path
import cv2
import config
import attendance
from face_engine import FaceEngine
from face_db import FaceDB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--part", required=True, choices=config.DEPARTMENTS)
    ap.add_argument("--images")
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()
    args.name = args.name.strip()
    if not args.name:
        ap.error("--name khong duoc de trong")
    if args.samples < 1:
        ap.error("--samples phai lon hon 0")

    engine, db, embs = FaceEngine(), FaceDB(), []

    if args.images:
        for p in sorted(Path(args.images).glob("*")):
            img = cv2.imread(str(p))
            face = engine.largest_face(img) if img is not None else None
            if face is not None:
                embs.append(engine.embedding(face))
    else:
        cap = cv2.VideoCapture(config.CAMERA_ID)
        last = 0
        while len(embs) < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            face = engine.largest_face(frame)
            view = frame.copy()
            if face is not None:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(view, f"{len(embs)}/{args.samples} - SPACE: chup, Q: thoat", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Enroll", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == 32 and face is not None and time.time() - last > 0.3:
                embs.append(engine.embedding(face))
                last = time.time()
        cap.release()
        cv2.destroyAllWindows()

    if not embs:
        print("Khong lay duoc mau khuon mat nao.")
        return
    db.replace(args.name, embs)
    db.save()
    attendance.register_person(args.name, args.part)
    print(
        f"Da dang ky '{args.name}' ({args.part}) voi {len(embs)} mau. "
        f"Tong nguoi: {len(db.people())}"
    )


if __name__ == "__main__":
    main()
