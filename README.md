# alertness-colab

特徴量CSVから分類モデル（`model.pkl`）を作る、Colabでも手元でも動く学習用リポジトリ。
本体（mediapipe等）に依存せず、`pandas` / `scikit-learn` だけで完結する。

入力は「特徴量CSV」だけ。CSVを作る側（動画→CSVの取り込み）は本体リポジトリ
（alertness-recognition の `ingest`）が担当し、こちらはその出力を学習するだけ。

## 使い方（手元）

```bash
pip install -r requirements.txt
python run_train.py               # 仮データを自動生成して通し確認（dummyアルゴリズム）
python run_train.py --algo rf     # アルゴリズムだけ差し替え
python run_train.py --data-dir path/to/csvs   # 実データ（取り込み済みCSV）で学習
```

実データがまだ無くても、`run_train.py` は仮データ（`make_dummy_data.py`）を生成して
「読み込み→学習→採点→保存」まで通す。

## 使い方（Colab）

```python
!git clone <このリポジトリのURL>
%cd alertness-colab
!pip install -q -r requirements.txt
```

その後 `train.ipynb` を開くか `run_train.py` を実行。実データは Google Drive を
マウントして `--data-dir` に渡すのが楽（顔動画・特徴量CSVは個人データなので repo に入れない）。

## 差し替えるのはここだけ

- **アルゴリズム**: `--algo`（または notebook の `ALGO` セル）で `dummy / logreg / svm / rf / voting` を切替。追加は `training/algorithms.py` に1関数足すだけ。
- **入力データ**: `--data-dir` を変えるだけ。CSVの列が本体の特徴量CSVと同じなら、どのデータセット由来でも動く。

## 構成

```
run_train.py          通しで学習を実行するエントリ
make_dummy_data.py    仮データ（合成特徴量CSV）の生成
train.ipynb           Colab用ノート（アルゴリズム差し替えセル付き）
requirements.txt      pandas / scikit-learn / joblib / numpy
training/
  config.py           特徴量列・ラベル列・アルゴリズム名など
  data.py             CSV読み込み → X / y / groups
  splits.py           被験者独立split
  algorithms.py       名前 → scikit-learn互換モデル（差し替え口）
  metrics.py          採点（本体 evaluation/metrics と同じ定義）
  artifact.py         model.pkl の保存/読み込み（特徴量列も同梱）
  pipeline.py         読み込み→学習→採点→保存の一連
```

## 出力

`model.pkl` は `{model, features, classes}` を1つに固めたもの。本体アプリはこの `features`
の順で特徴量を並べて `model.predict` を呼ぶ。データ・モデルは `.gitignore` 済み。
