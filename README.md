# He thong cham cong nhan dien khuon mat + chong gia mach

## Cai dat
    python -m venv venv && venv\Scripts\activate      # Windows
    python -m pip install -r requirements.txt
(Lan dau chay, InsightFace tu tai model ArcFace `buffalo_l` ~280MB.)
Python 3.10/3.11 la on dinh nhat voi mediapipe + insightface.

Neu da co thu muc `.venv`, run cac lenh bang Python trong moi truong nay:

    .\.venv\Scripts\python.exe enroll.py --name "Tony" --part BG6
    .\.venv\Scripts\python.exe main.py
    .\.venv\Scripts\python.exe main.py export

## Cac buoc
1. Dang ky nguoi dung:     `python enroll.py --name "Tony" --part BG6`
   (bo phan: BG6, BU7 hoac BU10; chup ~10 anh nhieu goc)
   Nguoi da dang ky tu truoc can dang ky lai voi `--part` de gan bo phan.
2. Thu du lieu CNN:        `python collect_data.py --label real`  roi  `--label spoof`
   (spoof = huong camera vao anh in, man hinh dien thoai/laptop phat anh/video; >=1500 anh moi lop)
3. Huan luyen CNN:         `python train_liveness.py --epochs 20`
4. Chay cham cong:         `python main.py`      (Q de thoat)
5. Xuat bao cao:           `python main.py export`   -> xuat file cho tat ca bo phan co cham cong trong ngay.
   Them ma bo phan de chi xuat bo phan do, vi du `python main.py export BG6`
   (ho tro BG6, BU7, BU10; file co anh trong cot photo).

## Luong xu ly
Camera -> InsightFace (detect + ArcFace 512-d) -> MediaPipe FaceMesh (EAR, goc xoay dau)
-> Thu thach NGAU NHIEN (xoay trai hoac phai, sau do nhin thang) -> CNN texture cham diem tung frame
-> Ket luan: CNN trung binh >= nguong VA hoan thanh thu thach VA >=60% frame cung 1 nguoi (FAISS)
-> Ghi SQLite (IN/OUT tu dong, chong ghi trung 60s), luu anh chinh dien tai `data/attendance_photos/`,
   hien thi ket qua tam thoi va quay lai trang thai san sang nhan dien tiep theo.

## Dieu chinh (config.py)
- RECOG_THRESHOLD: tang neu nhan nham nguoi la, giam neu hay khong nhan ra.
- LIVENESS_THRESHOLD, EAR_THRESH, TURN_*_RATIO: chinh theo camera/anh sang.

## Han che can biet
- Chua co model CNN thi chi con thu thach xoay dau/nhin thang (video quay san van co the lua neu
  khong dung huong ngau nhien; deepfake real-time can giai phap manh hon).
- Do chinh xac CNN phu thuoc du lieu ban tu thu; nen bo sung bo CASIA-FASD / Replay-Attack / CelebA-Spoof.
- Camera RGB don khong the chong 100% mat na 3D; he thong thuong mai dung them camera IR/depth.
