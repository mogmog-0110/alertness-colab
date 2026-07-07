"""仮のデータセット（合成特徴量CSV）を生成する。

まだ実データが無い段階で、学習の配線を通しで確認するためのもの。アプリの特徴量CSVと
同じ列を持ち、ラベルごとに特徴量の分布をずらしてあるので、dummy 以外の実アルゴリズムでも
学習できる。中身は乱数なので精度そのものに意味はない。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

_FIELDS = [
    "session_id", "subject", "timestamp", "face_present",
    "ear", "mar", "pitch_rel", "yaw_rel", "gaze_off", "jawOpen",
    "dim_drowsiness_score", "dim_drowsiness_level",
    "dim_distraction_score", "dim_distraction_level",
    "label",
]

# ラベルごとの特徴量の中心値（ずらしてあるので分類器が学習できる）。
_CENTERS = {
    "awake": {
        "ear": 0.32, "mar": 0.05, "pitch_rel": 0.0,
        "yaw_rel": 0.0, "gaze_off": 0.05, "jawOpen": 0.10,
    },
    "drowsiness": {
        "ear": 0.18, "mar": 0.30, "pitch_rel": 14.0,
        "yaw_rel": 0.0, "gaze_off": 0.06, "jawOpen": 0.45,
    },
    "distraction": {
        "ear": 0.30, "mar": 0.06, "pitch_rel": 2.0,
        "yaw_rel": 30.0, "gaze_off": 0.32, "jawOpen": 0.10,
    },
}
_SPREAD = {
    "ear": 0.02, "mar": 0.05, "pitch_rel": 4.0,
    "yaw_rel": 6.0, "gaze_off": 0.03, "jawOpen": 0.08,
}


def _rows_for_subject(subject: str, rng: np.random.Generator, per_class: int) -> list[dict]:
    rows: list[dict] = []
    t = 0.0
    for label, center in _CENTERS.items():
        for _ in range(per_class):
            values = {k: float(rng.normal(center[k], _SPREAD[k])) for k in center}
            values["ear"] = max(0.0, values["ear"])
            values["jawOpen"] = min(1.0, max(0.0, values["jawOpen"]))
            values["gaze_off"] = max(0.0, values["gaze_off"])
            rows.append({
                "session_id": subject,
                "subject": subject,
                "timestamp": round(t, 3),
                "face_present": 1,
                **{k: round(v, 4) for k, v in values.items()},
                "dim_drowsiness_score": 0.0,
                "dim_drowsiness_level": 0,
                "dim_distraction_score": 0.0,
                "dim_distraction_level": 0,
                "label": label,
            })
            t += 1 / 30.0
    return rows


def generate(out_dir: str, subjects: int = 3, per_class: int = 200, seed: int = 0) -> list[Path]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written: list[Path] = []
    for i in range(subjects):
        subject = f"dummy{i + 1:02d}"
        rows = _rows_for_subject(subject, rng, per_class)
        path = directory / f"{subject}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="仮の特徴量CSVを生成する")
    parser.add_argument("--out", default="sample_data", help="出力先ディレクトリ")
    parser.add_argument("--subjects", type=int, default=3)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    paths = generate(args.out, args.subjects, args.per_class, args.seed)
    print(f"{len(paths)} 件の仮データを生成: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
