"""学習済みモデルの保存/読み込み。軸ごとのモデル・特徴量列・クラスを一緒に固める。

軸ごとにモデルを持つ。アプリ側は推論時に features の順で特徴量を並べる必要が
あるので、モデル単体ではなく (models, features, classes) を1つのファイルに
まとめる。features には <列名>_present（値があったかのフラグ）も含まれる。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import joblib


def save_bundle(
    path: str,
    models: Mapping[str, object],
    features: Sequence[str],
    classes: Mapping[str, Sequence[str]],
) -> None:
    joblib.dump(
        {
            "models": dict(models),
            "features": list(features),
            "classes": {axis: list(values) for axis, values in classes.items()},
        },
        path,
    )


def load_bundle(path: str) -> dict:
    return joblib.load(path)
