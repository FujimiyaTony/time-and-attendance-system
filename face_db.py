"""Co so du lieu vector khuon mat bang FAISS (inner product tren vector da chuan hoa = cosine)."""
import json
import numpy as np
try:
    import faiss
except ImportError as exc:
    raise SystemExit(
        "Thieu thu vien FAISS. Hay kich hoat moi truong ao roi cai dat:\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  python -m pip install faiss-cpu"
    ) from exc
import config


class FaceDB:
    def __init__(self):
        self.index = faiss.IndexFlatIP(config.EMB_DIM)
        self.names: list[str] = []          # names[i] la chu cua vector thu i
        if config.FAISS_INDEX.exists() and config.NAMES_FILE.exists():
            try:
                self.index = faiss.read_index(str(config.FAISS_INDEX))
                names = json.loads(config.NAMES_FILE.read_text(encoding="utf-8"))
                if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                    raise ValueError("NAMES_FILE khong dung dinh dang danh sach ten.")
                if len(names) != self.index.ntotal:
                    raise ValueError(
                        f"Du lieu khong dong bo: index={self.index.ntotal}, names={len(names)}"
                    )
                self.names = names
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"Khong the tai co so du lieu khuon mat an toan: {exc}"
                ) from exc

    def add(self, name: str, embeddings):
        arr = np.asarray(embeddings, dtype="float32").reshape(-1, config.EMB_DIM)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        self.index.add(arr)
        self.names.extend([name] * len(arr))

    def replace(self, name: str, embeddings):
        """Thay the toan bo mau cu cua mot nguoi thay vi tao ban ghi trung ten."""
        arr = np.asarray(embeddings, dtype="float32").reshape(-1, config.EMB_DIM)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)

        keep_ids = [i for i, stored_name in enumerate(self.names) if stored_name != name]
        new_index = faiss.IndexFlatIP(config.EMB_DIM)
        if keep_ids:
            kept_vectors = np.asarray(
                [self.index.reconstruct(i) for i in keep_ids],
                dtype="float32",
            )
            new_index.add(kept_vectors)
        new_index.add(arr)
        self.index = new_index
        self.names = [self.names[i] for i in keep_ids] + [name] * len(arr)

    def search(self, emb):
        """Tra ve (ten, diem). Ten = None neu khong khop hoac DB rong."""
        if self.index.ntotal == 0:
            return None, 0.0
        q = np.asarray(emb, dtype="float32").reshape(1, -1)
        scores, ids = self.index.search(q, 1)
        score, idx = float(scores[0][0]), int(ids[0][0])
        if score < config.RECOG_THRESHOLD:
            return None, score
        return self.names[idx], score

    def people(self):
        return sorted(set(self.names))

    def save(self):
        if len(self.names) != self.index.ntotal:
            raise RuntimeError("Tu choi luu: FAISS index va danh sach ten khong dong bo.")

        config.FAISS_INDEX.parent.mkdir(parents=True, exist_ok=True)
        index_tmp = config.FAISS_INDEX.with_suffix(config.FAISS_INDEX.suffix + ".tmp")
        names_tmp = config.NAMES_FILE.with_suffix(config.NAMES_FILE.suffix + ".tmp")
        faiss.write_index(self.index, str(index_tmp))
        names_tmp.write_text(
            json.dumps(self.names, ensure_ascii=False), encoding="utf-8"
        )
        index_tmp.replace(config.FAISS_INDEX)
        names_tmp.replace(config.NAMES_FILE)
