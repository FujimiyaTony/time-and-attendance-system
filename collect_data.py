"""Thu thap du lieu huan luyen CNN chong gia mach.
   python collect_data.py --label real    # ban than that truoc camera
   python collect_data.py --label spoof   # huong camera vao anh in / man hinh dien thoai / video
Nhan Q de dung. Moi 0.2s luu 1 anh (da cat rong quanh mat).
Nen thu o nhieu anh sang, goc do, thiet bi; moi lop >= 1500 anh.
"""
import argparse
import time
import cv2
import config
from face_engine import FaceEngine
from liveness import crop_face


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["real", "spoof"], required=True)
    args = ap.parse_args()
    out = config.LIVENESS_DATASET / args.label
    out.mkdir(parents=True, exist_ok=True)

    engine = FaceEngine()
    cap = cv2.VideoCapture(config.CAMERA_ID)
    n, last = len(list(out.glob("*.jpg"))), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        face = engine.largest_face(frame)
        if face is not None and time.time() - last > 0.2:
            crop = crop_face(frame, face.bbox)
            if crop is not None:
                cv2.imwrite(str(out / f"{int(time.time()*1000)}.jpg"), crop)
                n += 1
                last = time.time()
        cv2.putText(frame, f"{args.label}: {n} anh (Q: thoat)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Collect", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
