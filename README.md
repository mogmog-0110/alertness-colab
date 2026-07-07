# alertness-colab

特徴量CSVから、眠気とよそ見（注意の逸れ）の判定モデル `model.pkl` を作る学習用リポジトリ。
本体（alertness-recognition）とは別物で、`pandas` / `scikit-learn` だけで動く（mediapipe不要）。
Colab で `train.ipynb` を開き、セルを上から実行するのが基本の使い方。

## セットアップ

```bash
pip install -r requirements.txt
```

Colab なら、クローンしてから最初のセルで `!pip install -q -r requirements.txt`。

## 全体の流れ

```
動画 ──[本体repoのingest]──▶ 特徴量CSV ──[このrepo]──▶ model.pkl ──▶ 本体アプリに置く
```

このリポジトリが担うのは真ん中（CSV → model.pkl）だけ。動画は扱わない。

## 使い方

`train.ipynb` を Colab（またはローカルの Jupyter）で開き、セルを上から順に実行する。
コマンドラインは使わない。差し替えるのはノート内の `ALGO` と `DATA_DIR` の2箇所だけ。

### 0. 動画 → CSV（前段。本体repo側）

学習の材料は「特徴量CSV」。これは本体 alertness-recognition が作る。2通り：

- 自前収録：アプリの `collect.bat` で、ラベル付きCSVが `runs/` に溜まる
- 公開データ：`python -m alertness.ingest --manifests <json/フォルダ>` で、動画→CSV

どちらも同じ列のCSVになる。ここ（colab）は、その出来上がったCSVを受け取るだけ。

### 1. CSV を置く

- 置き場所：`sample_data/`（既定）。別フォルダなら、ノートの `DATA_DIR` をそのパスに変える。
- 実データが無くても、`sample_data/` が空ならノートが仮データを自動生成して通し確認できる。
- CSVは「1行 = 1フレーム」。学習が読むのは次の列だけ（他の列は無視される）：

  | 列 | 意味 |
  |---|---|
  | `ear` `mar` `pitch_rel` `yaw_rel` `gaze_off` `jawOpen` | 特徴量（入力） |
  | `label_drowsiness` | 眠気の正解。`none` / `low` / `medium` / `high` |
  | `label_distraction` | よそ見の正解。`none` / `low` / `medium` / `high` |
  | `subject` | 被験者ID。同じ人が train と test に混ざらないよう分けるのに使う |

  例（一部の列）：

  ```
  subject,ear,mar,pitch_rel,yaw_rel,gaze_off,jawOpen,label_drowsiness,label_distraction
  s01,0.31,0.05,1.2,0.4,0.05,0.11,none,none
  s01,0.18,0.32,14.0,0.3,0.06,0.44,high,none
  ```

### 2. アルゴリズムを選ぶ

ノートの `ALGO` セルを書き換えるだけ。値は `algorithms/` にあるファイル名。

```python
ALGO = 'dummy'   # → algorithms/dummy.py が使われる
```

同梱は配線確認用の `dummy` だけ。実アルゴリズムは各自が足す（下記）。

### 3. アルゴリズムを足す

`algorithms/` に「1ファイル = 1アルゴリズム」で置く。ファイル名がそのまま `ALGO` の値になる。

1. `algorithms/_template.py` をコピーして `algorithms/名前.py` を作る
2. その `build()` に、scikit-learn 互換モデル（`.fit` / `.predict`）を返す処理を書く
3. ノートの `ALGO = '名前'` にして実行

```python
# algorithms/rf.py の例
from sklearn.ensemble import RandomForestClassifier

def build():
    return RandomForestClassifier(n_estimators=200, random_state=0)
```

`build()` は軸（眠気・よそ見）ごとに呼ばれるので、毎回新しいインスタンスを返すこと。
1ファイルずつ独立して足せるので、同じファイルを取り合わずに分担できる。

### 4. 学習してモデルを作る

ノートを上から実行すると、眠気・よそ見の2軸それぞれで1モデルを学習し、
軸ごとの採点表（accuracy・混同行列など）を表示して `model.pkl` を書き出す。
2軸のモデルと、使った特徴量の列順を1ファイルに同梱する。

### 5. モデルをアプリ側に置く

- できた `model.pkl` を本体 alertness-recognition の `models/` に置く。
- 本体は同梱された特徴量の列順どおりに並べて `predict` する。
- ※本体側でこの `model.pkl` を読み込む判定器（ルールベースと差し替えるML判定器）の配線は今後。
  いまはモデル生成まで。

## 構成

```
train.ipynb           学習ノート（これを実行する）
make_dummy_data.py    仮データ（合成特徴量CSV）の生成
requirements.txt      pandas / scikit-learn / joblib / numpy
algorithms/           1ファイル=1アルゴリズム（差し替え口）
  __init__.py         名前 → build() を呼ぶローダ
  _template.py        新規アルゴリズムのひな形（コピー元）
  dummy.py            配線確認用（多数派クラスを返すだけ）
training/
  config.py           特徴量列・ターゲット列など
  data.py             CSV読み込み → X / y / groups
  splits.py           被験者独立split
  metrics.py          採点（本体 evaluation/metrics と同じ定義）
  artifact.py         model.pkl の保存/読み込み（特徴量列も同梱）
  pipeline.py         読み込み→学習→採点→保存の一連
```

データ・モデルは `.gitignore` 済み（顔由来の特徴量CSVは個人データ、`*.pkl` は生成物）。
