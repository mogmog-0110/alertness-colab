"""学習の設定。特徴量列・ラベル列・アルゴリズム名などをここで一元管理する。"""

from __future__ import annotations

from dataclasses import dataclass, field

# アプリ推論時に出せる列だけを既定にする。学習と推論で列がズレると壊れるため。
DEFAULT_FEATURES = ("ear", "mar", "pitch_rel", "yaw_rel", "gaze_off", "jawOpen")


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "sample_data"
    out_path: str = "model.pkl"
    algorithm: str = "dummy"  # algorithms.build_model が解釈する名前
    label: str = "label"
    group: str = "subject"  # 被験者独立split に使う列
    features: tuple[str, ...] = field(default_factory=lambda: DEFAULT_FEATURES)
    test_size: float = 0.3
    seed: int = 0
