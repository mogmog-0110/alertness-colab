"""アルゴリズムの差し替え口。名前→ scikit-learn 互換モデル(.fit/.predict)を返す。

ここだけ差し替えれば学習・評価・保存は無改修。dummy はデータもアルゴリズムも
無い段階で配線確認するための最小実装（多数派クラスを返すだけ）。
アンサンブルも VotingClassifier で同じ口に乗る。
"""

from __future__ import annotations


def build_model(name: str):
    key = name.lower()

    if key == "dummy":
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent")

    if key == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000)

    if key == "svm":
        from sklearn.svm import SVC

        return SVC(probability=True)

    if key == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=200, random_state=0)

    if key == "voting":
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        return VotingClassifier(
            estimators=[
                ("logreg", LogisticRegression(max_iter=1000)),
                ("rf", RandomForestClassifier(n_estimators=200, random_state=0)),
                ("svm", SVC(probability=True)),
            ],
            voting="soft",
        )

    raise ValueError(f"未知のアルゴリズム: {name}（dummy/logreg/svm/rf/voting）")
