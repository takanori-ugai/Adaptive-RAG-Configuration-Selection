import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "../adaptive_rag_baseline.py",
]

BASE = Path(__file__).resolve().parent
CWD = BASE.parent


def main():
    print(f"Adaptive-RAG baseline: {sys.executable}")
    print(f"cwd: {CWD}")

    for script in SCRIPTS:
        path = (BASE / script).resolve()
        print(f"=== 実行中: {path} ===")
        subprocess.run([sys.executable, str(path)], check=True, cwd=str(CWD))
        print(f"=== 完了: {script} ===\n")


if __name__ == "__main__":
    main()
