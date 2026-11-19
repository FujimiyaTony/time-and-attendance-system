import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from openpyxl import load_workbook

import attendance
import config


class AttendanceExportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.patches = [
            patch.object(config, "BASE_DIR", self.root),
            patch.object(config, "DATA_DIR", self.root / "data"),
            patch.object(config, "DB_PATH", self.root / "data" / "attendance.db"),
            patch.object(
                config, "ATTENDANCE_PHOTO_DIR",
                self.root / "data" / "attendance_photos"),
            patch.object(config, "COOLDOWN_SECONDS", 0),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.frame = np.full((120, 160, 3), 127, dtype=np.uint8)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_log_saves_photo_and_export_embeds_it(self):
        attendance.register_person("Mina", "BG6")
        self.assertEqual(attendance.log("Mina", 0.91, self.frame), "IN")
        workbook_path = self.root / "attendance.xlsx"
        attendance.export_xlsx(workbook_path)

        workbook = load_workbook(workbook_path)
        sheet = workbook["Cham cong"]
        self.assertEqual(sheet["A2"].value, 1)
        self.assertEqual(sheet["B2"].value, "Mina")
        self.assertEqual(sheet["F1"].value, "photo")
        self.assertEqual(len(sheet._images), 1)

        with closing(sqlite3.connect(config.DB_PATH)) as connection:
            stored_path = connection.execute(
                "SELECT photo_path FROM attendance WHERE id=1"
            ).fetchone()[0]
        self.assertTrue((config.BASE_DIR / stored_path).is_file())

    def test_export_migrates_existing_database_without_photos(self):
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(config.DB_PATH)) as connection:
            with connection:
                connection.execute(
                    """CREATE TABLE attendance(
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           name TEXT NOT NULL, ts TEXT NOT NULL,
                           kind TEXT NOT NULL, score REAL)"""
                )
                connection.execute(
                    "INSERT INTO attendance(name, ts, kind, score) VALUES (?,?,?,?)",
                    ("Mina", "2026-11-18T08:00:00", "IN", 0.91),
                )

        workbook_path = self.root / "attendance.xlsx"
        attendance.export_xlsx(workbook_path)
        workbook = load_workbook(workbook_path)
        sheet = workbook["Cham cong"]

        self.assertIsNone(sheet["F2"].value)
        self.assertEqual(len(sheet._images), 0)

    def test_export_continues_when_photo_is_missing(self):
        attendance.register_person("Mina", "BG6")
        attendance.log("Mina", 0.91, self.frame)
        with closing(sqlite3.connect(config.DB_PATH)) as connection:
            photo_path = connection.execute(
                "SELECT photo_path FROM attendance WHERE id=1"
            ).fetchone()[0]
        (config.BASE_DIR / photo_path).unlink()

        workbook_path = self.root / "attendance.xlsx"
        attendance.export_xlsx(workbook_path)
        workbook = load_workbook(workbook_path)
        sheet = workbook["Cham cong"]
        self.assertEqual(sheet["F2"].value, "[anh khong ton tai]")
        self.assertEqual(len(sheet._images), 0)

    def test_malformed_timestamp_does_not_break_log(self):
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(config.DB_PATH)) as connection:
            with connection:
                connection.execute(
                    """CREATE TABLE attendance(
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           name TEXT NOT NULL, ts TEXT NOT NULL,
                           kind TEXT NOT NULL, score REAL, photo_path TEXT)"""
                )
                connection.execute(
                    "INSERT INTO attendance(name, ts, kind, score) VALUES (?,?,?,?)",
                    ("Mina", "khong-hop-le", "IN", 0.91),
                )
        attendance.register_person("Mina", "BG6")
        self.assertEqual(attendance.log("Mina", 0.95, self.frame), "IN")

    def test_log_crops_attendance_photo_around_face(self):
        attendance.register_person("Mina", "BG6")
        bbox = np.array([40, 20, 120, 100], dtype=np.float32)
        self.assertEqual(attendance.log("Mina", 0.91, self.frame, bbox), "IN")

        with closing(sqlite3.connect(config.DB_PATH)) as connection:
            photo_path = connection.execute(
                "SELECT photo_path FROM attendance WHERE id=1"
            ).fetchone()[0]
        stored_image = cv2.imread(str(config.BASE_DIR / photo_path))
        self.assertIsNotNone(stored_image)
        self.assertLess(stored_image.shape[1], self.frame.shape[1])

    def test_daily_export_splits_records_by_department(self):
        attendance.register_person("Mina", "BG6")
        attendance.register_person("Tony", "BU7")
        attendance.log("Mina", 0.91, self.frame)
        attendance.log("Tony", 0.92, self.frame)

        output_paths = attendance.export_xlsx(day=datetime.now().date())
        self.assertEqual(
            {path.name for path in output_paths},
            {
                f"attendance_BG6_{datetime.now():%Y-%m-%d}.xlsx",
                f"attendance_BU7_{datetime.now():%Y-%m-%d}.xlsx",
            },
        )
        bg6 = load_workbook(self.root / output_paths[0].name)
        bu7 = load_workbook(self.root / output_paths[1].name)
        names = {
            bg6["Cham cong"]["B2"].value,
            bu7["Cham cong"]["B2"].value,
        }
        self.assertEqual(names, {"Mina", "Tony"})

    def test_daily_export_can_select_one_department(self):
        attendance.register_person("Mina", "BG6")
        attendance.register_person("Tony", "BU7")
        attendance.log("Mina", 0.91, self.frame)
        attendance.log("Tony", 0.92, self.frame)

        output_paths = attendance.export_xlsx(
            day=datetime.now().date(), department="bg6"
        )
        self.assertEqual(len(output_paths), 1)
        self.assertTrue(output_paths[0].name.startswith("attendance_BG6_"))

        workbook = load_workbook(output_paths[0])
        sheet = workbook["Cham cong"]
        self.assertEqual(sheet["B2"].value, "Mina")
        self.assertIsNone(sheet["B3"].value)


if __name__ == "__main__":
    unittest.main()
