"""特徴量CSVの読み込みと、学習用の X / y / groups への変換。

前提は「アプリの特徴量CSVの列」だけ。どのデータセット由来かは問わない。
ラベル無しの行（label が空）は学習・評価から外す。
"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence

import pandas as pd


def load_frames(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"{data_dir} に CSV が見つかりません。")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def build_xy(
    df: pd.DataFrame, features: Sequence[str], label: str, group: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, list[str]]:
    if label not in df.columns:
        raise KeyError(f"ラベル列 '{label}' がCSVにありません。")

    labeled = df[df[label].astype(str).str.strip() != ""].copy()
    if labeled.empty:
        raise ValueError("ラベル付きの行が1件もありません。")

    cols = [c for c in features if c in labeled.columns]
    if not cols:
        raise KeyError(f"指定した特徴量列がCSVにありません: {list(features)}")

    x = labeled[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = labeled[label].astype(str)
    groups = labeled[group].astype(str) if group in labeled.columns else None
    return x, y, groups, cols
