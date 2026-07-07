"""学習の一連の流れ。読み込み→（軸ごとに）分割→学習→採点→保存を1本にまとめる。

正準ラベルは2軸なので、軸ごとに1モデルを学習する。ラベルが無い軸は飛ばすので、
眠気だけ付いたデータでも眠気モデルだけ学習できる。アルゴリズムは呼び出し側から
build_model として渡す（引数なしで新しいモデルを返す関数）。軸ごとに呼び直して
別インスタンスを得るので、1軸目の学習状態を2軸目に持ち越さない。
"""

from __future__ import annotations

from collections.abc import Callable

from .artifact import save_bundle
from .config import TrainConfig
from .data import build_xy, load_frames
from .metrics import scorecard
from .splits import split_indices


def _train_one(
    df, cfg: TrainConfig, target: str, build_model: Callable[[], object]
) -> tuple[object, dict, dict]:
    x, y, groups, cols = build_xy(df, cfg.features, target, cfg.group)
    train_idx, test_idx = split_indices(x, y, groups, cfg.test_size, cfg.seed)

    model = build_model()
    model.fit(x.iloc[train_idx], y.iloc[train_idx])

    y_true = list(y.iloc[test_idx])
    y_pred = list(model.predict(x.iloc[test_idx]))
    labels = sorted(set(y_true) | set(y_pred))
    score = scorecard(y_true, y_pred, labels, negative_label="none")
    return model, score, {"train": len(train_idx), "test": len(test_idx), "features": cols}


def run_training(cfg: TrainConfig, build_model: Callable[[], object]) -> dict:
    df = load_frames(cfg.data_dir)

    models: dict[str, object] = {}
    scores: dict[str, dict] = {}
    counts: dict[str, dict] = {}
    features: list[str] = []
    for target in cfg.targets:
        if target not in df.columns:
            continue
        model, score, info = _train_one(df, cfg, target, build_model)
        models[target] = model
        scores[target] = score
        counts[target] = {"train": info["train"], "test": info["test"]}
        features = info["features"]

    if not models:
        raise ValueError("学習できる軸がありません（ターゲット列が見つからない）。")

    classes = {axis: list(getattr(m, "classes_", [])) for axis, m in models.items()}
    save_bundle(cfg.out_path, models, features, classes)

    return {"scores": scores, "features": features, "counts": counts, "artifact": cfg.out_path}
