import csv
import logging
import os
import random
import re
import string
import time
import uuid
from itertools import product

import ollama
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

# Embeddings 抽象クラスの互換 import
try:
    from langchain_core.embeddings import Embeddings
except Exception:  # 古いLangChain向け
    from langchain.embeddings.base import Embeddings

logging.getLogger("datasets").setLevel(logging.WARNING)

# === 定数 ===
CSV_FILENAME = "../train_data/train_naiverag_squad.csv"
SAMPLE_NUM = 100

# === 選択肢リスト ===
embedding_model_list = ["nomic-embed-text", "mxbai-embed-large"]
llm_model_list = ["llama3.1:8b", "qwen3:14b"]
chunk_size_list = [250, 500]
chunk_overlap_list = [50, 100]
chain_type_list = ["stuff", "map_reduce"]


# === Ollama向けトークンカウンタ ===
def count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


# === Ollama用 Embeddings ラッパー（LangChain互換） ===
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # ★ 修正: embeddings() は input ではなく prompt を受け取る
        vectors: list[list[float]] = []
        for t in texts:
            res = ollama.embeddings(model=self.model, prompt=t)
            vectors.append(res["embedding"])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        res = ollama.embeddings(model=self.model, prompt=text)
        return res["embedding"]


# === データセット取得 ===
def prepare_dataset(sample_num: int = SAMPLE_NUM, seed: int = 42) -> tuple[list[dict], list[int]]:
    """SQuADの検証セットからランダムサンプルを抽出"""
    ds = load_dataset("rajpurkar/squad")
    # data = ds["validation"]
    data = ds["train"]
    random.seed(seed)
    rand_indices = random.sample(range(len(data)), sample_num)
    return data, rand_indices


# === 文書をベクトル化 ===
def document_embeddings_for_logging(
    text: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
):
    documents = [Document(page_content=doc.strip(), metadata={}) for doc in text.strip().split("\n\n") if doc.strip()]
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(documents)
    split_docs = [d if isinstance(d, Document) else Document(page_content=str(d)) for d in split_docs]

    # Ollama埋め込みをLangChainに渡す
    embeddings = OllamaEmbeddings(model=embedding_model)

    # ユニークなcollection_nameを毎回使う
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=f"tmp-{uuid.uuid4()}",
        persist_directory=None,
    )
    embedding_inputs = [d.page_content for d in split_docs]
    return vectordb, len(split_docs), embedding_inputs


# === LLM呼び出し（Ollama chat） ===
def ollama_chat(model: str, prompt: str, temperature: float = 0.0) -> str:
    res = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature} if temperature is not None else None,
    )
    return res["message"]["content"]


# === RAG QA ===
def custom_retrieval_qa_for_logging(
    question: str,
    vectordb: Chroma,
    top_k: int,
    llm_model: str,
    chain_type: str,
    start: float,
):
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    relevant_docs = retriever.invoke(question)
    relevant_docs = relevant_docs if isinstance(relevant_docs, list) else relevant_docs["documents"]
    chat_inputs: list[str] = []
    chat_outputs: list[str] = []

    if chain_type == "stuff":
        context = "\n".join([doc.page_content for doc in relevant_docs[:top_k]])
        prompt = (
            "Answer the following question using only the provided context.\n"
            f"Context:\n{context}\n"
            f"Question: {question}\n"
            "Output only answers in one sentence or some keywords."
        )
        result = ollama_chat(llm_model, prompt, temperature=0.0)
        end = time.time()
        chat_inputs.append(prompt)
        chat_outputs.append(result)

        run_time = end - start
        return prompt, normalize_answer(result), chat_inputs, chat_outputs, run_time

    if chain_type == "map_reduce":
        partial_answers = []
        for doc in relevant_docs[:top_k]:
            part_prompt = (
                "Answer the following question using only the provided context.\n"
                f"Context:\n{doc.page_content}\n"
                f"Question: {question}\n"
                "Output only answers in one sentence or some keywords."
            )
            part_result = ollama_chat(llm_model, part_prompt, temperature=0.0)
            partial_answers.append(part_result)
            chat_inputs.append(part_prompt)
            chat_outputs.append(part_result)

        reduce_prompt = (
            "Given the following partial answers, synthesize a final answer to the question:\n"
            "Partial Answers:\n" + "\n".join(partial_answers) + f"\nQuestion: {question}\n"
            "Output only answers in one sentence or some keywords."
        )
        result = ollama_chat(llm_model, reduce_prompt, temperature=0.0)
        end = time.time()
        chat_inputs.append(reduce_prompt)
        chat_outputs.append(result)

        run_time = end - start
        return reduce_prompt, normalize_answer(result), chat_inputs, chat_outputs, run_time

    raise NotImplementedError(f"chain_type '{chain_type}' is not implemented.")


# === 正規化 ===
def normalize_answer(s: str) -> str:
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


# === メイン ===
def main():
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
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, idx in enumerate(tqdm(indices, desc="Processing", ncols=100)):
            sample = data[idx]
            question = sample["question"]
            context = sample["context"]
            j = 0

            for embedding_model, llm_model, chunk_size, chunk_overlap, chain_type in product(
                embedding_model_list,
                llm_model_list,
                chunk_size_list,
                chunk_overlap_list,
                chain_type_list,
            ):
                print(
                    f"\n==== サンプル {i + 1}/{len(indices)} | パターン: {embedding_model}, {llm_model}, {chunk_size}, {chunk_overlap}, {chain_type} ====",
                )
                try:
                    start = time.time()
                    vectordb, num_chunks, embedding_inputs = document_embeddings_for_logging(
                        context + "\nj",
                        embedding_model,
                        chunk_size,
                        chunk_overlap,
                    )
                    j += 1
                    top_k = 4
                    actual_k = min(top_k, num_chunks)
                    prompt, answer, chat_inputs, chat_outputs, run_time = custom_retrieval_qa_for_logging(
                        question,
                        vectordb,
                        actual_k,
                        llm_model,
                        chain_type,
                        start,
                    )
                    true_answer = normalize_answer(str(sample["answers"]["text"][0]))
                    input_token = count_tokens(prompt)
                    output_token = count_tokens(answer)

                    # トークン数へ変換
                    input_embedding_tokens = [count_tokens(txt) for txt in embedding_inputs]
                    input_chat_tokens = [count_tokens(txt) for txt in chat_inputs]
                    output_chat_tokens = [count_tokens(txt) for txt in chat_outputs]

                    writer.writerow(
                        {
                            "id": sample["id"],
                            "time": run_time,
                            "question": question,
                            "answer": answer,
                            "true_answer": true_answer,
                            "rag_name": "naiverag",
                            "model_name": llm_model,
                            "embedding_model_name": embedding_model,
                            "chain_type": chain_type,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "input_token": input_token,
                            "output_token": output_token,
                            "context": context.replace("\n", ""),
                            "input_embedding": sum(input_embedding_tokens),
                            "input_chat": sum(input_chat_tokens),
                            "output_chat": sum(output_chat_tokens),
                        },
                    )
                    # print("== 質問 ==\n", question)
                    # print("== 回答 ==\n", answer)
                    # print("== プロンプト ==\n", prompt)
                except Exception as e:
                    print(
                        f"Error at param pattern: {embedding_model}, {llm_model}, {chunk_size}, {chunk_overlap}, {chain_type} | {e}",
                    )


if __name__ == "__main__":
    main()
