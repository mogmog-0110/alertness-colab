"""配線確認用。学習せず多数派クラスを返すだけ。実データ・実アルゴリズムが無くても通せる。"""

from sklearn.dummy import DummyClassifier


def build():
    return DummyClassifier(strategy="most_frequent")
