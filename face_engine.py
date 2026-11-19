"""Phat hien mat + trich xuat vector ArcFace (InsightFace)."""
import numpy as np
import config


class FaceEngine:
    def __init__(self):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=config.DET_SIZE)

    def largest_face(self, frame_bgr):
        faces = self.app.get(frame_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    @staticmethod
    def embedding(face) -> np.ndarray:
        return face.normed_embedding.astype("float32")   # da L2-normalize, 512-d
