# alertness-colab

特徴量CSVから、眠気とよそ見（注意の逸れ）の判定モデル `model.pkl` を作る学習用リポジトリ。
本体（alertness-recognition）とは別物で、`pandas` / `scikit-learn` だけで動く（mediapipe不要）。
Colab で `train.ipynb` を開き、セルを上から実行するのが基本の使い方。

## セットアップ

Colab で `train.ipynb` を開いて、上のセルから順に実行するだけ。最初のセルがリポジトリの取得
（`git clone`）と依存インストールをやる。Colab は .ipynb だけ読み込むので、コード本体を clone
しないと `import` が通らない点に注意（そのセルが面倒を見る）。

ローカルの Jupyter で動かすなら、クローンしたフォルダの中で依存を入れておく。

```bash
pip install -r requirements.txt
```

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
  | `ear_norm` `mar_rel` `pitch_rel` `yaw_rel` `gaze_off` | 特徴量（幾何。キャリブで個人差を吸収した相対値） |
  | `jawOpen` `eyeBlinkLeft` `eyeBlinkRight` | 特徴量（あくび・瞬き） |
  | `browDown*`(AU4) `eyeSquint*`(AU7) `mouthPress*`(AU24) `mouthFrown*`(AU15) `cheekSquint*`(AU6) `browInnerUp` | 特徴量（表情。ストレスの手がかり） |
  | `hr_bpm` `rppg_quality` `hrv_rmssd` | 特徴量（rPPG。**大半のフレームで空になる**） |

  rPPG の列は信号品質のしきい値を通ったフレームでしか埋まらない（実測で2割ほど）。
  学習では欠損を 0 で埋めるだけにせず `<列名>_present` を自動で足し、「値があったか」も
  特徴として渡す。アプリ側の推論も同じ規約で埋めるので、列の並びは `model.pkl` の
  `features` が唯一の取り決めになる。
  | `label_drowsiness` | 眠気の正解。`none` / `low` / `medium` / `high` |
  | `label_distraction` | よそ見の正解。`none` / `low` / `medium` / `high` |
  | `label_concentration` | 集中の正解。`none` / `low` / `medium` / `high`（任意） |
  | `label_stress` | ストレスの正解。`none` / `low` / `medium` / `high`（任意・生体信号由来） |
  | `subject` | 被験者ID。同じ人が train と test に混ざらないよう分けるのに使う |
  | `context` | 用途タグ（study / driving 等）。用途別にモデルを分けるとき使う（任意） |

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

### 用途(context)で分ける（任意）

注意逸脱（distraction）の意味は用途で変わる（運転の脇見と自習の脇見は別物）。
ノートの `CONTEXT` に用途名を入れると、注意逸脱はその用途のフレームだけで学習し、
`model_study.pkl` のように用途別のモデルを書き出す。

- `CONTEXT = ''`：全用途をプールして1モデル（既定）
- `CONTEXT = 'study'`：注意逸脱は study のフレームだけで学習

眠気（drowsiness）は用途で意味が変わらないので、`CONTEXT` を指定しても常に全データで
学習する（`training/config.py` の `CONTEXT_FREE_AXES`）。用途を分けるほど1モデルあたりの
データは減るため、データが少ないうちは `CONTEXT=''` で始めるのがよい。

用途別モデルはアプリ側の config で `policy.model_path: models/model_study.pkl` のように
用途に合ったファイルを指すだけで使える。

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
