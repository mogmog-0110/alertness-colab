"""学習の設定。特徴量列・ターゲット列などをここで一元管理する。"""

from __future__ import annotations

from dataclasses import dataclass, field

# アプリ推論時に出せる列だけを既定にする。学習と推論で列がズレると壊れるため。
DEFAULT_FEATURES = ("ear", "mar", "pitch_rel", "yaw_rel", "gaze_off", "jawOpen")

# 正準ラベルは2軸。軸ごとに1モデルを学習する。
DEFAULT_TARGETS = ("label_drowsiness", "label_distraction")


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "sample_data"
    out_path: str = "model.pkl"
    targets: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TARGETS)
    group: str = "subject"  # 被験者独立split に使う列
    features: tuple[str, ...] = field(default_factory=lambda: DEFAULT_FEATURES)
    test_size: float = 0.3
    seed: int = 0
