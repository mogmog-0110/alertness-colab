"""SVM（RBFカーネル）によるアルゴリズム実装。

## 前処理: RobustScaler
特徴量6列（ear, mar, pitch_rel, yaw_rel, gaze_off, jawOpen）はスケールも分布の裾の
重さもバラバラ（例: mar/jawOpen はあくび等で稀に大きく跳ねる）。SVMは距離ベースなので
スケーリングは必須だが、あくびや横向きなどの極端値はノイズではなくラベルの根拠その
ものであることが多いため、平均・分散で正規化する StandardScaler より、中央値・IQRを
使う RobustScaler の方が少数の極端値にスケールを引っ張られにくく、値自体も削らずに
残せる。

## カーネル: RBF
EAR がある閾値を下回ると急に「眠気: high」になる、といった非線形の境界を想定し RBF を
採用。特徴量が6次元・データもさほど大きくない想定なので計算コストは問題にならない。

## class_weight="balanced"
平常時（none）が多数派になるのはほぼ確実なので、不均衡対策として指定する。

## ハイパーパラメータ（C, gamma）を固定値にしている理由
training/pipeline.py の _train_one() は model.fit(X, y) しか呼ばず、被験者ID（groups）
を渡してくれない。そのため build() の中で GridSearchCV(cv=StratifiedGroupKFold(...))
のようなグループ考慮のチューニングを安全に組み込めない（非グループの
StratifiedKFold で代用すると、同じ被験者のフレームが学習・検証の両方に混ざり、
精度を過大評価する形でリークする）。

そのため、C・gamma のチューニングはこのリポジトリの外（別スクリプトやノート）で
実データを使い、StratifiedGroupKFold で被験者単位に区切ったグリッドサーチを行い、
その結果をここに定数として反映する運用を想定している。下の SEARCH_SPACE / チューニ
ング手順のコメントを参照。

build() は軸（眠気・よそ見）ごとに呼ばれるので、毎回新しいインスタンスを返す。
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

# オフラインでチューニングした結果をここに反映する。
# 実データが揃うまでの暫定値（オーダーだけ合わせた値。要チューニング）。
C = 4.0
GAMMA = "scale"


def build():
    return Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=GAMMA,
                    class_weight="balanced",
                    decision_function_shape="ovr",
                    probability=False,  # 確率が要る運用になったら True に切り替える
                    random_state=0,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# オフラインでのハイパーパラメータ探索の例（このリポジトリのノートからは呼ばない）。
# 実データが揃ったら、被験者IDを groups として渡し、下記のように別途実行して
# 良かった C / GAMMA を上の定数に反映する運用を想定。
# ---------------------------------------------------------------------------
def _tune_offline(x, y, groups):
    """例: python標準の実行ではなく、必要になったときに手元で呼ぶための参考実装。"""
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("svc", SVC(kernel="rbf", class_weight="balanced", random_state=0)),
        ]
    )
    param_grid = {
        "svc__C": [0.5, 1, 2, 4, 8, 16],
        "svc__gamma": ["scale", 0.01, 0.05, 0.1, 0.5],
    }
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    search.fit(x, y, groups=groups)
    return search.best_params_, search.best_score_