"""Keep one checkpoint loaded so probe photos skip the reload."""

import threading
from pathlib import Path

from open_set import OpenSetError, load_scorer, score_loaded


class ProbeScorer:
    def __init__(self):
        self._lock = threading.Lock()
        self._key = None
        self._loaded = None

    def score(self, model_path, bank_dir, image_paths):
        model_path = Path(model_path)
        bank_dir = Path(bank_dir)
        npz = bank_dir / "bank.npz"
        meta = bank_dir / "meta.json"
        if not npz.is_file() or not meta.is_file():
            raise OpenSetError(f"No open-set bank in {bank_dir}.")
        key = (
            str(model_path.resolve()),
            model_path.stat().st_mtime_ns,
            model_path.stat().st_size,
            npz.stat().st_mtime_ns,
            meta.stat().st_mtime_ns,
        )
        with self._lock:
            if key != self._key or self._loaded is None:
                self._loaded = load_scorer(model_path, bank_dir)
                self._key = key
            model, device, idx_to_class, bank = self._loaded
            return score_loaded(
                model, device, idx_to_class, bank, image_paths
            )
