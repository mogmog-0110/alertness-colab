"""学習アルゴリズムは、このフォルダに「1つ1ファイル」で置く。ファイル名が選択名になる。

各ファイルは build() を定義し、scikit-learn 互換モデル（.fit / .predict）を返す。
_template.py をコピーして中身を書けば、ノートブックの ALGO からそのまま選べる。
1人1ファイルで足せるので、同じファイルを取り合わずに作業分担できる。
"""

from importlib import import_module


def get(name: str):
    try:
        module = import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"アルゴリズム '{name}' が algorithms/ にありません。") from exc
    if not hasattr(module, "build"):
        raise AttributeError(f"algorithms/{name}.py に build() がありません。")
    return module.build()
