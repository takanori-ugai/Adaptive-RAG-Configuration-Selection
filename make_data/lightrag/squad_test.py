import asyncio
import csv
import os
import random
import re
import shutil
import string
import time
from itertools import product

import networkx as nx
from datasets import load_dataset
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc, setup_logger
from pyvis.network import Network

# === 定数・セットアップ ===
CSV_FILENAME = "../test_data/test_lightrag_squad.csv"
SAMPLE_NUM = 20
WORKING_DIR = "./rag_storage"
DATA_FOLDER = "../kg_squad_test"
RANDOM_SEED = 42

embedding_model_list = ["nomic-embed-text", "mxbai-embed-large"]
llm_model_list = ["llama3.1:8b", "qwen3:14b"]
chunk_size_list = [500, 1000]
chunk_overlap_list = [100, 250]
retrieval_type_list = ["global", "local"]


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


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


# === データセットロード ===
def prepare_dataset(sample_num=SAMPLE_NUM, seed=RANDOM_SEED):
    ds = load_dataset("rajpurkar/squad")
    # data = ds["train"]
    data = ds["validation"]
    random.seed(seed)
    rand_indices = random.sample(range(len(data)), sample_num)
    return data, rand_indices


# === LightRAG初期化 ===
async def initialize_rag(embedding_model, llm_model, chunk_size, chunk_overlap):
    if embedding_model == "nomic-embed-text":
        rag = LightRAG(
            working_dir=WORKING_DIR,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                func=lambda texts: ollama_embed(texts, embed_model="nomic-embed-text"),
            ),
            llm_model_func=ollama_model_complete,
            llm_model_name=llm_model,
            chunk_token_size=chunk_size,
            chunk_overlap_token_size=chunk_overlap,
        )
    elif embedding_model == "mxbai-embed-large":
        rag = LightRAG(
            working_dir=WORKING_DIR,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                func=lambda texts: ollama_embed(texts, embed_model="mxbai-embed-large"),
            ),
            llm_model_func=ollama_model_complete,
            llm_model_name=llm_model,
            chunk_token_size=chunk_size,
            chunk_overlap_token_size=chunk_overlap,
        )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def prepare_working_dir():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)


async def process_sample(
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
    rag = None
    try:
        start = time.time()
        rag = await initialize_rag(embedding_model, llm_model, chunk_size, chunk_overlap)

        await rag.ainsert(context)
        prompt = f"Question: {question}\nOutput only answers in one sentence or some keywords."
        answer = await rag.aquery(
            prompt,
            param=QueryParam(mode=retrieval_type),
        )
        end = time.time()
        answer = normalize_answer(answer)
        true_answer = normalize_answer(str(sample["answers"]["text"][0]))
        input_token = count_tokens(prompt)
        output_token = count_tokens(answer)
        run_time = end - start

        # プロンプト・出力のトークン数
        input_embedding_tokens = count_tokens(context)
        input_chat_tokens = count_tokens(prompt) + count_tokens(context)
        output_chat_tokens = count_tokens(answer)

        writer.writerow(
            {
                "id": sample["id"],
                "time": run_time,
                "question": question,
                "answer": answer,
                "true_answer": true_answer,
                "rag_name": "lightrag",
                "model_name": llm_model,
                "embedding_model_name": embedding_model,
                "chain_type": retrieval_type,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "input_token": input_token,
                "output_token": output_token,
                "context": context.replace("\n", ""),
                "input_embedding": input_embedding_tokens,
                "input_chat": input_chat_tokens,
                "output_chat": output_chat_tokens,
                "index": j,
            },
        )
        print(f"== 質問 ==\n{question}\n== 回答 ==\n{answer}\n== プロンプト ==\n{prompt}")

        graphml_path = f"{WORKING_DIR}/graph_chunk_entity_relation.graphml"
        if not os.path.exists(graphml_path):
            print(f"[INFO] 空グラフを書き出します: {graphml_path}")
            G_empty = nx.Graph()
            nx.write_graphml(G_empty, graphml_path)
        G = nx.read_graphml(graphml_path)
        net = Network(notebook=True)
        net.from_nx(G)
        net.show(f"{WORKING_DIR}/knowledge_graph.html")

        # グラフデータのコピー
        new_folder_path = f"{DATA_FOLDER}/{sample['id']}_{j}"
        print("\n= フォルダ名 =\n", new_folder_path)
        # フォルダが存在する場合は削除
        if os.path.exists(new_folder_path):
            shutil.rmtree(new_folder_path)

        shutil.copytree(WORKING_DIR, new_folder_path)

    except Exception as e:
        print(f"An error occurred (ID: {sample['id']}): {e}")

    finally:
        if rag:
            await rag.finalize_storages()


async def main():
    data, indices = prepare_dataset()
    os.makedirs(os.path.dirname(CSV_FILENAME), exist_ok=True)
    with open(CSV_FILENAME, mode="w", encoding="utf-8", newline="") as csvfile:
        fieldnames = [
            "id",
            "time",
            "question",
            "answer",
            "true_answer",
            "rag_name",
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
            context = sample["context"]

            # 全組み合わせループ
            j = 0
            for embedding_model, llm_model, chunk_size, chunk_overlap, retrieval_type in product(
                embedding_model_list,
                llm_model_list,
                chunk_size_list,
                chunk_overlap_list,
                retrieval_type_list,
            ):
                context_addnum = context + f"\n{j}"
                setup_logger("lightrag", level="INFO")
                prepare_working_dir()
                print(
                    f"\n==== サンプル {i + 1}/{len(indices)} | パターン: {embedding_model}, {llm_model}, {chunk_size}, {chunk_overlap}, {retrieval_type} ====",
                )
                try:
                    await process_sample(
                        j,
                        writer,
                        sample,
                        question,
                        context_addnum,
                        embedding_model,
                        llm_model,
                        chunk_size,
                        chunk_overlap,
                        retrieval_type,
                    )
                    j += 1

                    # if j > 1:
                    #     break
                except Exception as e:
                    print(
                        f"Error at param pattern: {embedding_model}, {llm_model}, {chunk_size}, {chunk_overlap}, {retrieval_type} | {e}",
                    )
                # break


if __name__ == "__main__":
    asyncio.run(main())
