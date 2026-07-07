"""学習済みモデルの保存/読み込み。特徴量列とクラスも一緒に固める。

アプリ側は推論時にこの features の順で特徴量を並べる必要があるので、
モデル単体ではなく (model, features, classes) を1つのファイルにまとめる。
"""

from __future__ import annotations

from collections.abc import Sequence

import joblib


def save_bundle(path: str, model, features: Sequence[str], classes: Sequence[str]) -> None:
    joblib.dump(
        {"model": model, "features": list(features), "classes": list(classes)}, path
    )


def load_bundle(path: str) -> dict:
    return joblib.load(path)
