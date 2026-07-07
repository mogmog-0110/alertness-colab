"""学習の一連の流れ。読み込み→分割→学習→採点→保存を1本にまとめる。

アルゴリズムやデータが変わっても、この流れ自体は変えない。差し替えるのは
config.algorithm（＝algorithms.build_model の名前）とデータの中身だけ。
"""

from __future__ import annotations

from .algorithms import build_model
from .artifact import save_bundle
from .config import TrainConfig
from .data import build_xy, load_frames
from .metrics import scorecard
from .splits import split_indices


def run_training(cfg: TrainConfig) -> dict:
    df = load_frames(cfg.data_dir)
    x, y, groups, cols = build_xy(df, cfg.features, cfg.label, cfg.group)
    train_idx, test_idx = split_indices(x, y, groups, cfg.test_size, cfg.seed)

    model = build_model(cfg.algorithm)
    model.fit(x.iloc[train_idx], y.iloc[train_idx])

    y_true = list(y.iloc[test_idx])
    y_pred = list(model.predict(x.iloc[test_idx]))
    labels = sorted(set(y_true) | set(y_pred))
    score = scorecard(y_true, y_pred, labels)

    classes = list(getattr(model, "classes_", labels))
    save_bundle(cfg.out_path, model, cols, classes)

    return {
        "score": score,
        "features": cols,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "artifact": cfg.out_path,
    }
