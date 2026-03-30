"""translate - optimized by autoprogramming."""
import json
import importlib
from datetime import datetime, timezone
from pathlib import Path

from .schema import French

_root = Path(__file__).parent
_scores = json.loads((_root / "scores.json").read_text())
_active = max(
    (k for k in _scores if k != "active"),
    key=lambda k: _scores[k].get("val_avg", 0),
)
_module = importlib.import_module(f".candidates.{_active}", package=__name__)

predict = _module.predict


class _Program:
    """Callable wrapper that feels like a plain function, with optional logging."""

    def __init__(self):
        self.logging = False
        self._log_path = _root / "logs"

    def __call__(self, english: str) -> French:
        result = predict(english)
        if self.logging:
            self._save(english, result)
        return result

    def _save(self, english: str, result: French):
        self._log_path.mkdir(exist_ok=True)
        log_file = self._log_path / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
        entry = json.dumps({
            "english": english,
            "French": str(result),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        with open(log_file, "a") as f:
            f.write(entry + "\n")

    def enable_logging(self):
        self.logging = True
        return self

    def disable_logging(self):
        self.logging = False
        return self


translate = _Program()
