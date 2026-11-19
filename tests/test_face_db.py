import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import config
from face_db import FaceDB


class FaceDBTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.patches = [
            patch.object(config, "DATA_DIR", self.root / "data"),
            patch.object(config, "FAISS_INDEX", self.root / "data" / "faces.index"),
            patch.object(config, "NAMES_FILE", self.root / "data" / "names.json"),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_replace_removes_old_embeddings_for_same_name(self):
        db = FaceDB()
        db.add("Mina", [np.eye(config.EMB_DIM, dtype="float32")[0]])
        db.add("Tony", [np.eye(config.EMB_DIM, dtype="float32")[1]])
        db.replace("Mina", [np.eye(config.EMB_DIM, dtype="float32")[2]])
        db.save()

        reloaded = FaceDB()
        self.assertEqual(reloaded.names.count("Mina"), 1)
        self.assertEqual(reloaded.names.count("Tony"), 1)

    def test_loading_rejects_index_names_mismatch(self):
        db = FaceDB()
        db.add("Mina", [np.eye(config.EMB_DIM, dtype="float32")[0]])
        db.save()
        config.NAMES_FILE.write_text(json.dumps(["Tony", "Extra"]), encoding="utf-8")

        with self.assertRaises(RuntimeError):
            FaceDB()
