import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "../../train/pipeline/make_model.py",  # 実データ生成 + ランキングモデル構築
    "../adaptive_rag_baseline.py",  # 実データに対する Adaptive-RAG ベースライン評価
]

BASE = Path(__file__).resolve().parent
CWD = BASE.parent.parent


def main():
    print(f"Adaptive-RAG real-data pipeline: {sys.executable}")
    print(f"cwd: {CWD}")

    for script in SCRIPTS:
        path = (BASE / script).resolve()
        print(f"=== 実行中: {path} ===")
        subprocess.run([sys.executable, str(path)], check=True, cwd=str(CWD))
        print(f"=== 完了: {script} ===\n")


if __name__ == "__main__":
    main()
