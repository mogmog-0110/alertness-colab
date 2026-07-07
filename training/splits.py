"""学習/テストの分割。被験者が分かるなら被験者独立split にする。

同じ人がtrainとtestに混ざると精度が甘く出るので、group（subject）があれば
人ごとに分ける。無ければ通常のランダム分割にフォールバックする。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_indices(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if groups is not None and groups.nunique() > 1:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(x, y, groups))
        return train_idx, test_idx

    from sklearn.model_selection import train_test_split

    stratify = y if y.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(
        np.arange(len(x)), test_size=test_size, random_state=seed, stratify=stratify
    )
    return train_idx, test_idx
