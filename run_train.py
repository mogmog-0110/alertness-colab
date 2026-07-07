"""学習をコマンド1つで通しで動かすエントリ。

実データが無ければ仮データを生成してから学習するので、そのまま実行すれば
「読み込み→学習→採点→model.pkl 保存」の配線がすべて確認できる。

例:
    python run_train.py                      # 仮データ＋dummy で通し確認
    python run_train.py --algo rf            # アルゴリズムだけ差し替え
    python run_train.py --data-dir runs/ing  # 実データ（取り込み済みCSV）で学習
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_dummy_data
from training.config import DEFAULT_FEATURES, TrainConfig
from training.metrics import format_scorecard
from training.pipeline import run_training


def _has_csv(data_dir: str) -> bool:
    if not os.path.isdir(data_dir):
        return False
    for _root, _dirs, files in os.walk(data_dir):
        if any(f.endswith(".csv") for f in files):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="特徴量CSVから分類モデルを学習する")
    parser.add_argument("--data-dir", default="sample_data")
    parser.add_argument("--algo", default="dummy", help="dummy/logreg/svm/rf/voting")
    parser.add_argument("--out", default="model.pkl")
    parser.add_argument("--no-dummy", action="store_true", help="仮データの自動生成をしない")
    args = parser.parse_args(argv)

    if not _has_csv(args.data_dir):
        if args.no_dummy:
            print(f"{args.data_dir} にCSVがありません。")
            return 1
        print(f"{args.data_dir} が空なので仮データを生成します。")
        make_dummy_data.generate(args.data_dir)

    cfg = TrainConfig(
        data_dir=args.data_dir, out_path=args.out, algorithm=args.algo, features=DEFAULT_FEATURES
    )
    result = run_training(cfg)

    print(f"\ntrain={result['n_train']}  test={result['n_test']}  features={result['features']}")
    print(format_scorecard(result["score"]))
    print(f"\nsaved: {result['artifact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
