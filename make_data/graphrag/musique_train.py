import csv
import os
import random
import re
import string
import subprocess
import time
from itertools import product

from datasets import load_dataset
from ruamel.yaml import YAML

# === 定数・セットアップ ===
CSV_FILENAME = "./train_data/train_graphrag_musique.csv"
SAMPLE_NUM = 100
RANDOM_SEED = 42

embedding_model_list = ["nomic-embed-text", "mxbai-embed-large"]
llm_model_list = ["llama3.1:8b", "qwen3:14b"]
chunk_size_list = [500, 1000]
chunk_overlap_list = [100, 250]
retrieval_type_list = ["local", "global"]

pattern = re.compile(r"graphrag\.index\.verbs\.text\.embed\.strategies\.openai INFO embedding\s+(\d+)\s+inputs via")


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


# === normalize_answer ===
def normalize_answer(s):
    def lower(text):
        return text.lower()

    def remove_punctuation(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))


# === データセットロード ===
def context_to_string(context):
    if isinstance(context, list):
        parts = []
        for idx, item in enumerate(context):
            title = f"Title_{idx + 1}"
            sents = []
            if isinstance(item, dict):
                title = item.get("title", title)
                sents = item.get("sentences", [])
            else:
                # 想定: [title, [sentences...]]
                try:
                    title, sents = item
                except Exception:
                    sents = item

            if not isinstance(sents, list):
                sents = [str(sents)]
            text = " ".join(map(str, sents))
            parts.append(f"{title}\n{text}")
        return "\n\n".join(parts)

    # 辞書形式（保険）
    if isinstance(context, dict):
        sentences = context.get("sentences", [])
        titles = context.get("title", [])
        result = []
        for idx, sents in enumerate(sentences):
            title = titles[idx] if idx < len(titles) else f"Title_{idx + 1}"
            text = " ".join(sents) if isinstance(sents, list) else str(sents)
            result.append(f"{title}\n{text}")
        return "\n\n".join(result)

    # どれでもなければ文字列化
    return str(context)


# === データセットロード ===
def prepare_dataset(sample_num=SAMPLE_NUM, seed=RANDOM_SEED):
    ds = load_dataset("dgslibisey/MuSiQue")
    # data = ds["validation"]
    data = ds["train"]
    random.seed(seed)
    rand_indices = random.sample(range(len(data)), sample_num)
    return data, rand_indices


# === データセットロード ===
def adjustment_yaml(embedding_model, llm_model, chunk_size, chunk_overlap):
    # YAMLオブジェクトを作成
    yaml = YAML()
    # 書式を保持するように設定
    yaml.preserve_quotes = True
    # YAMLファイルを読み込む
    with open("../settings.yaml") as file:
        data = yaml.load(file)
    data["llm"]["model"] = llm_model
    data["embeddings"]["llm"]["model"] = embedding_model
    data["chunks"]["size"] = chunk_size
    data["chunks"]["overlap"] = chunk_overlap
    # 変更内容を同じファイルに書き込む
    with open("../settings.yaml", "w") as file:
        yaml.dump(data, file)
    print("settings.yamlの書き換え完了")


def read_text(file_name):
    with open(file_name, encoding="utf-8") as f:
        doc = f.read()
    return doc


def entity_count(log_file):
    numbers = []

    with open(log_file, encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                numbers.append(int(match.group(1)))
    return numbers


def prompt_tuning(command):
    subprocess.run(command, check=False)


def run_graphrag(
    i,
    j,
    writer,
    sample,
    question,
    context,
    embedding_model,
    llm_model,
    chunk_size,
    chunk_overlap,
    retrieval_type,
):
    context2 = context + f"\n{j}"
    try:
        start = time.time()
        # embedding_model = "nomic-embed-text"
        # llm_model = "qwen3:14b"
        # chunk_size = 500
        # chunk_overlap = 100
        # retrieval_type = "global"
        adjustment_yaml(embedding_model, llm_model, chunk_size, chunk_overlap)
        ### 入力テキストファイルに書き込み
        with open("../input/book.txt", "w", encoding="utf-8") as f:
            f.write(context2)

        ### コマンドを用いて知識グラフを作成
        command = ["python", "-m", "graphrag.index", "--root", ".."]
        subprocess.run(command, check=False)

        # クエリ実行
        query_q = "Output only answers in one sentence or some keywords. " + question
        command = ["python", "-m", "graphrag.query", "--root", "..", "--method", retrieval_type, query_q]
        # command = ["../.venv/bin/graphrag", "query", "--root", "..", "--method", "global", "--query", query_q]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        end = time.time()
        run_time = end - start

        # 回答の正規化
        # print("===回答==\n", result)
        answer = ""
        target_label = "Local" if retrieval_type == "local" else "Global"
        match = re.search(rf"{target_label} Search Response:(.*)", result.stdout.replace("\n", ""))
        answer = match.group(1).strip() if match else result.stdout.replace("\n", "")

        # 評価用データの準備
        answer = normalize_answer(answer)
        gold = sample.get("answer")
        if isinstance(gold, (list, tuple)):
            gold = gold[0] if gold else ""
        true_answer = normalize_answer(str(gold))
        input_token = count_tokens(query_q)
        output_token = count_tokens(answer)

        # サブフォルダ一覧を取得
        folders = [f for f in os.listdir("../output") if os.path.isdir(os.path.join("../output", f))]
        latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join("../output", f)))

        log_file = f"../output/{latest_folder}/reports/indexing-engine.log"
        number = entity_count(log_file)

        embedding_tokens = count_tokens(context) * 2 + input_token
        input_tokens = (
            count_tokens(read_text("../prompts/entity_extraction.txt")) * number[0]
            + count_tokens(read_text("../prompts/summarize_descriptions.txt")) * number[0]
            + count_tokens(read_text("../prompts/community_report.txt")) * number[0]
            + count_tokens(read_text("../prompts/community_report.txt")) * number[0]
        )
        output_tokens = count_tokens(context) + output_token
        sample_id = sample.get("id", sample.get("_id", i))

        writer.writerow(
            {
                "id": sample.get("id", sample.get("_id", i)),
                "rag_name": "graphrag",
                "time": run_time,
                "question": question,
                "answer": answer,
                "true_answer": true_answer,
                "model_name": llm_model,
                "embedding_model_name": embedding_model,
                "chain_type": retrieval_type,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "input_token": input_token,
                "output_token": output_token,
                "context": context.replace("\n", ""),
                "input_embedding": embedding_tokens,
                "input_chat": input_tokens,
                "output_chat": output_tokens,
                "index": j,
            },
        )

    except Exception as e:
        print(
            f"Error at param pattern: {embedding_model}, {llm_model}, {chunk_size}, {chunk_overlap}, {retrieval_type} | {e}",
        )
        writer.writerow(
            {
                "id": sample["id"],
                "rag_name": "graphrag",
                "time": 0,
                "question": question,
                "answer": "",
                "true_answer": true_answer,
                "model_name": llm_model,
                "embedding_model_name": embedding_model,
                "chain_type": retrieval_type,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "input_token": 0,
                "output_token": 0,
                "context": context.replace("\n", ""),
                "input_embedding": 0,
                "input_chat": 0,
                "output_chat": 0,
                "index": j,
            },
        )


def main():
    data, indices = prepare_dataset()
    os.makedirs(os.path.dirname(CSV_FILENAME), exist_ok=True)
    with open(CSV_FILENAME, mode="w", encoding="utf-8", newline="") as csvfile:
        fieldnames = [
            "id",
            "rag_name",
            "time",
            "question",
            "answer",
            "true_answer",
            "model_name",
            "embedding_model_name",
            "chain_type",
            "chunk_size",
            "chunk_overlap",
            "input_token",
            "output_token",
            "context",
            "input_embedding",
            "input_chat",
            "output_chat",
            "index",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, idx in enumerate(indices):
            sample = data[idx]
            question = sample["question"]
            context = context_to_string(sample["paragraphs"])
            # question = "What took the place of Mac's Xserve server?"
            # context = "In 1998, after the return of Steve Jobs, Apple consolidated its multiple consumer-level desktop models into the all-in-one iMac G3, which became a commercial success and revitalized the brand. Since their transition to Intel processors in 2006, the complete lineup is entirely based on said processors and associated systems. Its current lineup comprises three desktops (the all-in-one iMac, entry-level Mac mini, and the Mac Pro tower graphics workstation), and four laptops (the MacBook, MacBook Air, MacBook Pro, and MacBook Pro with Retina display). Its Xserve server was discontinued in 2011 in favor of the Mac Mini and Mac Pro."

            # 全組み合わせループ
            j = 0
            for embedding_model, llm_model, chunk_size, chunk_overlap, retrieval_type in product(
                embedding_model_list,
                llm_model_list,
                chunk_size_list,
                chunk_overlap_list,
                retrieval_type_list,
            ):
                print(
                    f"\n==== サンプル {i + 1}/{len(indices)} | パターン: {embedding_model}, {llm_model}, {chunk_size}, {chunk_overlap}, {retrieval_type} ====",
                )
                run_graphrag(
                    i,
                    j,
                    writer,
                    sample,
                    question,
                    context,
                    embedding_model,
                    llm_model,
                    chunk_size,
                    chunk_overlap,
                    retrieval_type,
                )
                j += 1


if __name__ == "__main__":
    main()
