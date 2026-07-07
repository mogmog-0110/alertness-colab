"""仮のデータセット（合成特徴量CSV）を生成する。

まだ実データが無い段階で、学習の配線を通しで確認するためのもの。アプリの特徴量CSVと
同じ列を持ち、2軸（drowsiness / distraction）それぞれで段階(none/low/medium/high)ごとに
特徴量の分布をずらしてある。中身は乱数なので精度そのものに意味はない。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_FIELDS = [
    "session_id", "subject", "timestamp", "face_present",
    "ear", "mar", "pitch_rel", "yaw_rel", "gaze_off", "jawOpen",
    "dim_drowsiness_score", "dim_drowsiness_level",
    "dim_distraction_score", "dim_distraction_level",
    "label", "label_drowsiness", "label_distraction", "context",
]

_LEVELS = ["none", "low", "medium", "high"]
_SPREAD = {
    "ear": 0.02, "mar": 0.03, "pitch_rel": 3.0,
    "yaw_rel": 4.0, "gaze_off": 0.03, "jawOpen": 0.05,
}


def _noisy(center: dict, rng: np.random.Generator) -> dict:
    values = {k: float(rng.normal(center[k], _SPREAD[k])) for k in center}
    values["ear"] = max(0.0, values["ear"])
    values["gaze_off"] = max(0.0, values["gaze_off"])
    values["jawOpen"] = min(1.0, max(0.0, values["jawOpen"]))
    return values


def _drowsiness_center(k: int) -> dict:
    # 段階が上がるほど 目が閉じ・あくび・うつむきが増える。
    return {
        "ear": 0.32 - 0.05 * k, "mar": 0.05 + 0.08 * k, "pitch_rel": 5.0 * k,
        "yaw_rel": 0.0, "gaze_off": 0.05, "jawOpen": 0.10 + 0.12 * k,
    }


def _distraction_center(k: int) -> dict:
    # 段階が上がるほど 視線外れ・横向きが増える。
    return {
        "ear": 0.32, "mar": 0.05, "pitch_rel": 0.0,
        "yaw_rel": 12.0 * k, "gaze_off": 0.05 + 0.10 * k, "jawOpen": 0.10,
    }


def _row(
    subject: str, context: str, t: float, values: dict, drowsiness: str, distraction: str
) -> dict:
    return {
        "session_id": subject, "subject": subject, "timestamp": round(t, 3), "face_present": 1,
        **{k: round(v, 4) for k, v in values.items()},
        "dim_drowsiness_score": "", "dim_drowsiness_level": "",
        "dim_distraction_score": "", "dim_distraction_level": "",
        "label": "", "label_drowsiness": drowsiness, "label_distraction": distraction,
        "context": context,
    }


def _rows_for_subject(
    subject: str, context: str, rng: np.random.Generator, per_level: int
) -> list[dict]:
    rows: list[dict] = []
    t = 0.0
    for k, level in enumerate(_LEVELS):
        for _ in range(per_level):
            values = _noisy(_drowsiness_center(k), rng)
            rows.append(_row(subject, context, t, values, level, "none"))
            t += 1 / 30.0
    for k, level in enumerate(_LEVELS):
        for _ in range(per_level):
            values = _noisy(_distraction_center(k), rng)
            rows.append(_row(subject, context, t, values, "none", level))
            t += 1 / 30.0
    return rows


def generate(
    out_dir: str,
    subjects: int = 4,
    per_level: int = 120,
    seed: int = 0,
    contexts: tuple[str, ...] = ("study", "driving"),
) -> list[Path]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written: list[Path] = []
    for i in range(subjects):
        subject = f"dummy{i + 1:02d}"
        context = contexts[i % len(contexts)]  # 用途を被験者ごとに振り分ける
        rows = _rows_for_subject(subject, context, rng, per_level)
        path = directory / f"{subject}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    return written
