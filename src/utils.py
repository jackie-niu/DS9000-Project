from __future__ import annotations
import json
from pathlib import Path
import joblib
from typing import Any, Dict, Optional


# Model I/O
def save_model(model: Any, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    return str(p)


# Metrics storage
def append_metrics_jsonl(record: Dict[str, Any], path: str = "models/metrics.jsonl") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return str(p)


def load_metrics_jsonl(path: str = "models/metrics.jsonl") -> list[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# Metrics leaderboard
def metrics_leaderboard(path: str = "models/metrics.jsonl",
                        sort_key: str = "test_metrics.average_precision",
                        reverse: bool = True,
                        top: Optional[int] = 10) -> list[Dict[str, Any]]:

    def get_nested(d: Dict[str, Any], dotted: str) -> Any:
        cur = d
        for k in dotted.split("."):
            cur = cur.get(k, None) if isinstance(cur, dict) else None
        return cur

    records = load_metrics_jsonl(path)
    records.sort(key=lambda r: (get_nested(r, sort_key) is None, get_nested(r, sort_key)), reverse=reverse)
    return records[:top] if top else records

