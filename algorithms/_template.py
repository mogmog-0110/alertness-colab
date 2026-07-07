"""アルゴリズムのテンプレート。このファイルをコピーして名前を付け、build() を書く。

ファイル名がノートブックで選ぶ名前になる（例: myalgorithm.py なら ALGO = "myalgorithm"）。
build() は scikit-learn 互換のモデル（.fit / .predict を持つ）を返すこと。

build() は軸（眠気・よそ見）ごとに呼ばれるので、毎回「新しいインスタンス」を返す。
モデルを使い回すと2軸目の学習が1軸目の状態を引きずる。
学習の入力 X は数値の特徴量列、正解 y は4段階(none/low/medium/high)の多クラス。
"""


def build():
    # 例: from sklearn.ensemble import RandomForestClassifier
    #     return RandomForestClassifier(n_estimators=200, random_state=0)
    raise NotImplementedError("build() を実装してください。")
