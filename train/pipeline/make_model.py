import subprocess
import sys
from pathlib import Path

# 実行スクリプト
SCRIPTS = [
    "../make_train_data.py",  # 訓練データ構築
    "../make_test_data.py",  # テストデータ構築
    "../make_ranking_model.py",  # ランキングモデル構築
]

BASE = Path(__file__).resolve().parent
CWD = BASE.parent


def main():
    print(f"ランキングモデル構築と推論：{sys.executable}")
    print(f"cwd: {CWD}")

    for script in SCRIPTS:
        path = (BASE / script).resolve()
        print(f"=== 実行中: {path} ===")
        subprocess.run([sys.executable, str(path)], check=True, cwd=str(CWD))
        print(f"=== 完了: {script} ===\n")


if __name__ == "__main__":
    main()
