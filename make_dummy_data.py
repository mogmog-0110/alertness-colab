"""仮のデータセット（合成特徴量CSV）を生成する。

まだ実データが無い段階で、学習の配線を通しで確認するためのもの。アプリの特徴量CSVと
同じ列を持ち、4軸（drowsiness / distraction / concentration / stress）それぞれで
段階(none/low/medium/high)ごとに特徴量の分布をずらしてある。中身は乱数なので精度そのものに
意味はない。

rPPG 由来の列（hr_bpm / rppg_quality / hrv_rmssd）は実運用でも大半のフレームで欠ける
（信号品質のしきい値を通らない）ので、ここでも意図的に欠損させる。欠損の扱いまで含めて
配線を確認するため。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# アプリが記録する特徴量の列（training/config.py の DEFAULT_FEATURES と揃える）。
_GEOMETRY = ("ear_norm", "mar_rel", "pitch_rel", "yaw_rel", "gaze_off")
_BLINK = ("jawOpen", "eyeBlinkLeft", "eyeBlinkRight")
_ACTION_UNITS = (
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "eyeSquintLeft",
    "eyeSquintRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "cheekSquintLeft",
    "cheekSquintRight",
)
_RPPG = ("hr_bpm", "rppg_quality", "hrv_rmssd")
_FEATURES = _GEOMETRY + _BLINK + _ACTION_UNITS + _RPPG

_AXES = ("drowsiness", "distraction", "concentration", "stress")
_FIELDS = (
    ["session_id", "subject", "timestamp", "face_present"]
    + list(_FEATURES)
    + [f"dim_{axis}_{kind}" for axis in _AXES for kind in ("score", "level")]
    + ["label"]
    + [f"label_{axis}" for axis in _AXES]
    + ["context"]
)

_LEVELS = ["none", "low", "medium", "high"]
_SPREAD = {
    "ear_norm": 0.06,
    "mar_rel": 0.03,
    "pitch_rel": 3.0,
    "yaw_rel": 4.0,
    "gaze_off": 0.02,
    "jawOpen": 0.05,
    "eyeBlinkLeft": 0.05,
    "eyeBlinkRight": 0.05,
    "hr_bpm": 2.5,
    "rppg_quality": 0.08,
    "hrv_rmssd": 6.0,
}
_AU_SPREAD = 0.04
_RPPG_PRESENT = 0.2  # rPPG の値が出るフレームの割合（実測に合わせている）

# 何も起きていないときの中立値。段階ごとにここからずらす。
_NEUTRAL = {
    "ear_norm": 1.0,
    "mar_rel": 0.02,
    "pitch_rel": 0.0,
    "yaw_rel": 0.0,
    "gaze_off": 0.01,
    "jawOpen": 0.08,
    "eyeBlinkLeft": 0.1,
    "eyeBlinkRight": 0.1,
    "hr_bpm": 68.0,
    "rppg_quality": 0.7,
    "hrv_rmssd": 45.0,
    **dict.fromkeys(_ACTION_UNITS, 0.1),
}


def _center(axis: str, k: int) -> dict:
    """軸と段階から、その状態らしい特徴量の中心値を作る。"""
    center = dict(_NEUTRAL)
    if axis == "drowsiness":
        # 目が閉じ、あくびが増え、うつむく。
        center["ear_norm"] = 1.0 - 0.22 * k
        center["mar_rel"] = 0.02 + 0.08 * k
        center["jawOpen"] = 0.08 + 0.12 * k
        center["pitch_rel"] = 5.0 * k
        center["eyeBlinkLeft"] = center["eyeBlinkRight"] = 0.1 + 0.2 * k
    elif axis == "distraction":
        # 視線が外れ、横を向く。
        center["yaw_rel"] = 12.0 * k
        center["gaze_off"] = 0.01 + 0.10 * k
    elif axis == "concentration":
        # 段階が上がるほど「集中している」＝視線が載り頭部が安定する。
        center["gaze_off"] = 0.06 - 0.016 * k
        center["yaw_rel"] = 12.0 - 3.5 * k
    elif axis == "stress":
        # 心拍が上がり HRV が下がる。眉とまぶたが緊張する（AU4/AU7）。
        center["hr_bpm"] = 68.0 + 7.0 * k
        center["hrv_rmssd"] = 45.0 - 8.0 * k
        for name in ("browDownLeft", "browDownRight", "eyeSquintLeft", "eyeSquintRight"):
            center[name] = 0.1 + 0.12 * k
    return center


def _noisy(center: dict, rng: np.random.Generator) -> dict:
    values = {k: float(rng.normal(v, _SPREAD.get(k, _AU_SPREAD))) for k, v in center.items()}
    for name in ("ear_norm", "gaze_off", "hr_bpm", "rppg_quality", "hrv_rmssd"):
        values[name] = max(0.0, values[name])
    for name in _BLINK + _ACTION_UNITS + ("rppg_quality",):
        values[name] = min(1.0, max(0.0, values[name]))
    # rPPG は大半のフレームで欠ける。欠損は空文字（CSV上の空欄）で表す。
    if rng.random() > _RPPG_PRESENT:
        for name in _RPPG:
            values[name] = None
    return values


def _row(subject: str, context: str, t: float, values: dict, labels: dict) -> dict:
    row = {
        "session_id": subject,
        "subject": subject,
        "timestamp": round(t, 3),
        "face_present": 1,
        "label": "",
        "context": context,
    }
    for name, value in values.items():
        row[name] = "" if value is None else round(value, 4)
    for axis in _AXES:
        row[f"dim_{axis}_score"] = ""
        row[f"dim_{axis}_level"] = ""
        row[f"label_{axis}"] = labels.get(axis, "none")
    return row


def _rows_for_subject(
    subject: str, context: str, rng: np.random.Generator, per_level: int
) -> list[dict]:
    rows: list[dict] = []
    t = 0.0
    for axis in _AXES:
        for k, level in enumerate(_LEVELS):
            for _ in range(per_level):
                rows.append(_row(subject, context, t, _noisy(_center(axis, k), rng), {axis: level}))
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
