import subprocess
import sys
import time
from pathlib import Path

question = "Who wrote The Old Man and the Sea?"
context = "The Old Man and the Sea is a short novel written by Ernest Hemingway in 1951"
features_json = "../artifacts/results/one_qd_features.json"

W_PRE, W_REC, W_F1, W_TIME, W_TOKEN = 0.00, 0.00, 0.50, 0.50, 0.00
atk = 5

# 実行スクリプト
SCRIPTS = [
    "../one_feature_extraction.py",  # 特徴量抽出
    "../one_selection_rag.py",  # RAG構成選択
]

BASE = Path(__file__).resolve().parent
CWD = BASE.parent


def main():
    print(f"提案手法の試実験：{sys.executable}")
    print(f"cwd: {CWD}")
    start = time.time()

    middle_time = None

    for script in SCRIPTS:
        path = (BASE / script).resolve()
        cmd = [sys.executable, str(path)]
        print(f"=== 実行中: {path} ===")

        if path.name == "one_feature_extraction.py":
            cmd += ["--question", question, "--context", context, "--features_json", features_json]
            subprocess.run(cmd, check=True, cwd=str(CWD))
            middle_time = time.time()
            print(f"=== 完了: {script} ===\n")

        elif path.name == "one_selection_rag.py":
            cmd += [
                "--question",
                question,
                "--context",
                context,
                "--features_json",
                features_json,
                "--w_pre",
                str(W_PRE),
                "--w_rec",
                str(W_REC),
                "--w_f1",
                str(W_F1),
                "--w_time",
                str(W_TIME),
                "--w_token",
                str(W_TOKEN),
                "--atk",
                str(atk),
            ]
            subprocess.run(cmd, check=True, cwd=str(CWD))
            print(f"=== 完了: {script} ===\n")
        else:
            subprocess.run(cmd, check=True, cwd=str(CWD))
            print(f"=== 完了: {script} ===\n")

    end = time.time()

    if middle_time is None:
        middle_time = start

    print("☆")
    print(f"特徴抽出時間：{middle_time - start:.2f} 秒")
    print(f"RAG構成選択時間：{end - middle_time:.2f} 秒")
    print(f"全体実行時間: {end - start:.2f} 秒")


if __name__ == "__main__":
    main()
