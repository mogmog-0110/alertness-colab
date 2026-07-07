"""学習の設定。特徴量列・ターゲット列などをここで一元管理する。"""

from __future__ import annotations

from dataclasses import dataclass, field

# アプリ推論時に出せる列だけを既定にする。学習と推論で列がズレると壊れるため。
DEFAULT_FEATURES = ("ear", "mar", "pitch_rel", "yaw_rel", "gaze_off", "jawOpen")

# 正準ラベルは2軸。軸ごとに1モデルを学習する。
DEFAULT_TARGETS = ("label_drowsiness", "label_distraction")

# 用途(context)に依存しない軸。眠気は運転でも自習でも「眠い」は同じなので、
# context を指定しても常に全データで学習する。用途で意味が変わるのは注意逸脱だけ。
CONTEXT_FREE_AXES = ("label_drowsiness",)


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "sample_data"
    out_path: str = "model.pkl"
    targets: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TARGETS)
    group: str = "subject"  # 被験者独立split に使う列
    features: tuple[str, ...] = field(default_factory=lambda: DEFAULT_FEATURES)
    # 空なら全用途をプール。"study" 等にすると用途依存の軸だけその用途で学習する。
    context: str = ""
    context_free_axes: tuple[str, ...] = field(default_factory=lambda: CONTEXT_FREE_AXES)
    test_size: float = 0.3
    seed: int = 0
