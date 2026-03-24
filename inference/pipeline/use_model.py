import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "../use_ranking_model.py",  # ランキングモデルの予測
    "../selection_rag.py",  # ランキング結果に基づくRAG構成選択
]

BASE = Path(__file__).resolve().parent
CWD = BASE.parent


def main():
    print(f"RAG構成選択：{sys.executable}")
    print(f"cwd: {CWD}")

    for script in SCRIPTS:
        path = (BASE / script).resolve()
        print(f"=== 実行中: {path} ===")
        subprocess.run([sys.executable, str(path)], check=True, cwd=str(CWD))
        print(f"=== 完了: {script} ===\n")


if __name__ == "__main__":
    main()
