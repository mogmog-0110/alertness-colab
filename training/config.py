"""学習の設定。特徴量列・ターゲット列などをここで一元管理する。"""

from __future__ import annotations

from dataclasses import dataclass, field

# 欠けている列に付ける「値があったか」フラグの接尾辞。アプリ側の推論も同じ規約に従う。
PRESENT_SUFFIX = "_present"

# アプリ推論時に出せる列だけを既定にする。学習と推論で列がズレると壊れるため。
# 幾何はキャリブで個人差を吸収した相対値を使う（生の pitch/yaw は ±180 を跨いで
# 折り返すので学習に入れない）。normalize_version は取り違え防止の目印で特徴ではない。
DEFAULT_FEATURES = (
    # 幾何（目・口・頭部・視線）
    "ear_norm",
    "mar_rel",
    "pitch_rel",
    "yaw_rel",
    "gaze_off",
    # 瞬き・あくび
    "jawOpen",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    # 表情。FACS の AU に対応する blendshape で、browDown=AU4, eyeSquint=AU7,
    # mouthPress=AU24, mouthFrown=AU15, cheekSquint=AU6。ストレスの手がかりになる。
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
    # rPPG（カメラからの心拍）。OPTIONAL_FEATURES にも入れてある。
    "hr_bpm",
    "rppg_quality",
    "hrv_rmssd",
)

# 欠けるのが普通の列。rPPG は信号品質のしきい値を通らないと値が出ず、実測では
# 全フレームの2割ほどしか埋まらない。0 で埋めるだけだと「心拍0bpm」という実在しない
# 値を学習してしまうので、<列名>_present を足して「欠けていた」ことも特徴として渡す。
OPTIONAL_FEATURES = ("hr_bpm", "rppg_quality", "hrv_rmssd")

# 正準ラベルは4軸。軸ごとに1モデルを学習する。CSVに無い軸は自動で飛ばすので、
# 一部の軸しか持たないデータでも該当軸だけ学習できる。
DEFAULT_TARGETS = (
    "label_drowsiness",
    "label_distraction",
    "label_concentration",
    "label_stress",
)

# 用途(context)に依存しない軸。眠気・ストレスは運転でも自習でも意味が変わらないので、
# context を指定しても常に全データで学習する。用途で意味が変わるのは注意逸脱と集中。
CONTEXT_FREE_AXES = ("label_drowsiness", "label_stress")


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "sample_data"
    out_path: str = "model.pkl"
    targets: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TARGETS)
    group: str = "subject"  # 被験者独立split に使う列
    features: tuple[str, ...] = field(default_factory=lambda: DEFAULT_FEATURES)
    # 欠損があってよい列。欠けていたかどうかを別の特徴として足す。
    optional_features: tuple[str, ...] = field(default_factory=lambda: OPTIONAL_FEATURES)
    # 空なら全用途をプール。"study" 等にすると用途依存の軸だけその用途で学習する。
    context: str = ""
    context_free_axes: tuple[str, ...] = field(default_factory=lambda: CONTEXT_FREE_AXES)
    test_size: float = 0.3
    seed: int = 0
