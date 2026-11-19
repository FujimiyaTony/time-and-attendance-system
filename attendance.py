"""Ghi nhan cham cong vao SQLite + xuat Excel."""
import csv
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage

import config


def _conn():
    c = sqlite3.connect(config.DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS attendance(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT NOT NULL, ts TEXT NOT NULL, kind TEXT NOT NULL,
                   score REAL, photo_path TEXT, department TEXT)""")
    columns = {row[1] for row in c.execute("PRAGMA table_info(attendance)")}
    if "photo_path" not in columns:
        c.execute("ALTER TABLE attendance ADD COLUMN photo_path TEXT")
    if "department" not in columns:
        c.execute("ALTER TABLE attendance ADD COLUMN department TEXT")
    c.execute("""CREATE TABLE IF NOT EXISTS people(
                   name TEXT PRIMARY KEY, department TEXT NOT NULL)""")
    c.commit()
    return c


def register_person(name: str, department: str):
    name = name.strip()
    department = department.strip().upper()
    if not name:
        raise ValueError("Ten nguoi dung khong duoc de trong.")
    if department not in config.DEPARTMENTS:
        raise ValueError(
            f"Bo phan khong hop le. Chi chap nhan: {', '.join(config.DEPARTMENTS)}"
        )
    with closing(_conn()) as c, c:
        c.execute(
            """INSERT INTO people(name, department) VALUES (?, ?)
               ON CONFLICT(name) DO UPDATE SET department=excluded.department""",
            (name, department),
        )
        c.execute(
            """UPDATE attendance SET department=?
               WHERE name=? AND department IS NULL""",
            (department, name),
        )


def _attendance_photo(frame: np.ndarray, bbox=None) -> np.ndarray:
    """Cat anh chinh dien quanh khuon mat; giu frame day du neu khong co bbox."""
    if bbox is None:
        return frame

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_w, face_h = x2 - x1, y2 - y1
    pad = max(face_w, face_h) * 0.18
    cx1 = max(0, int(x1 - pad))
    cy1 = max(0, int(y1 - pad))
    cx2 = min(w, int(x2 + pad))
    cy2 = min(h, int(y2 + pad))
    crop = frame[cy1:cy2, cx1:cx2]
    return crop if crop.size else frame


def log(name: str, score: float, frame: np.ndarray, bbox=None) -> str | None:
    """Ghi vao/ra. Tra ve 'IN'/'OUT', hoac None neu dang trong thoi gian cho (cooldown)."""
    now = datetime.now()
    with closing(_conn()) as c:
        c.execute("BEGIN IMMEDIATE")
        person = c.execute(
            "SELECT department FROM people WHERE name=?", (name,)
        ).fetchone()
        if person is None:
            c.rollback()
            raise ValueError(
                f"Chua gan bo phan cho {name}. Hay dang ky lai voi --part."
            )
        department = person[0]
        last = c.execute(
            "SELECT ts, kind FROM attendance WHERE name=? ORDER BY id DESC LIMIT 1",
            (name,)
        ).fetchone()
        if last:
            try:
                last_time = datetime.fromisoformat(last[0])
            except (TypeError, ValueError):
                last = None
                last_time = None
            if last_time is not None and (
                now - last_time
            ).total_seconds() < config.COOLDOWN_SECONDS:
                c.rollback()
                return None
        kind = "OUT" if (
            last and last_time is not None and last[1] == "IN"
            and last[0][:10] == now.date().isoformat()
        ) else "IN"

        photo_frame = _attendance_photo(frame, bbox)
        encoded_ok, encoded = cv2.imencode(
            ".jpg", photo_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not encoded_ok:
            raise OSError("Khong the ma hoa anh cham cong.")

        config.ATTENDANCE_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
        photo_path = config.ATTENDANCE_PHOTO_DIR / f"{uuid4().hex}.jpg"
        relative_photo_path = photo_path.relative_to(config.BASE_DIR).as_posix()
        try:
            photo_path.write_bytes(encoded.tobytes())
            with c:
                c.execute(
                    """INSERT INTO attendance(name, ts, kind, score, photo_path, department)
                       VALUES (?,?,?,?,?,?)""",
                    (name, now.isoformat(timespec="seconds"), kind,
                     round(score, 4), relative_photo_path, department))
        except (OSError, sqlite3.Error):
            photo_path.unlink(missing_ok=True)
            c.rollback()
            raise
        c.commit()
        return kind


def export_csv(path="attendance.csv"):
    with closing(_conn()) as c, open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "timestamp", "kind", "score"])
        w.writerows(c.execute("SELECT id, name, ts, kind, score FROM attendance ORDER BY id"))
    print("Da xuat:", path)


def _write_xlsx(rows, path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Cham cong"
    sheet.append(["id", "name", "timestamp", "kind", "score", "photo"])
    sheet.column_dimensions["F"].width = 20

    for row_number, row in enumerate(rows, start=2):
        sheet.append([*row[:5], ""])
        sheet.row_dimensions[row_number].height = 70
        if not row[5]:
            continue

        photo_path = Path(row[5])
        if not photo_path.is_absolute():
            photo_path = config.BASE_DIR / photo_path
        if not photo_path.is_file():
            sheet.cell(row=row_number, column=6, value="[anh khong ton tai]")
            continue

        image = ExcelImage(str(photo_path))
        scale = min(120 / image.width, 90 / image.height, 1)
        image.width = int(image.width * scale)
        image.height = int(image.height * scale)
        sheet.add_image(image, f"F{row_number}")

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    workbook.save(path)


def export_xlsx(path=None, day=None, department=None):
    """Khong truyen path: xuat workbook theo bo phan trong ngay duoc chon."""
    if path is not None:
        with closing(_conn()) as c:
            rows = c.execute(
                "SELECT id, name, ts, kind, score, photo_path FROM attendance ORDER BY id"
            ).fetchall()
        _write_xlsx(rows, path)
        print("Da xuat:", path)
        return [Path(path)]

    if department is not None:
        department = department.strip().upper()
        if department not in config.DEPARTMENTS:
            raise ValueError(
                f"Bo phan khong hop le. Chi chap nhan: {', '.join(config.DEPARTMENTS)}"
            )
    export_day = day or datetime.now().date()
    day_text = export_day.isoformat() if hasattr(export_day, "isoformat") else str(export_day)
    with closing(_conn()) as c:
        if department:
            departments = [(department,)]
        else:
            departments = c.execute(
                """SELECT DISTINCT department FROM attendance
                   WHERE substr(ts, 1, 10)=? AND department IS NOT NULL
                   ORDER BY department""",
                (day_text,),
            ).fetchall()
        output_paths = []
        for (department,) in departments:
            rows = c.execute(
                """SELECT id, name, ts, kind, score, photo_path
                   FROM attendance
                   WHERE substr(ts, 1, 10)=? AND department=?
                   ORDER BY id""",
                (day_text, department),
            ).fetchall()
            output_path = config.BASE_DIR / f"attendance_{department}_{day_text}.xlsx"
            _write_xlsx(rows, output_path)
            output_paths.append(output_path)
            print("Da xuat:", output_path)
    if not output_paths:
        print(f"Khong co du lieu cham cong cho ngay {day_text}.")
    return output_paths
