# 各 RAG の実行結果から訓練データセットを構築
# 特徴量抽出結果を中間データに保存し，データセットの種類ごとに分類
# 提案手法で使用しなかった特徴量もある

from __future__ import annotations

import os
from collections.abc import Iterable
from functools import lru_cache

import gensim
import gensim.downloader as gensim_api
import numpy as np
import pandas as pd
import spacy
import torch
import yake
from bert_score.scorer import BERTScorer
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# 特徴量抽出実行の有無（実行は一回で良いため）
flag = True  # True にすると analyze_and_save() を実行

# 入力パス（各RAGで構築した生データ）
INPUT_CSV_FILES: list[str] = [
    "../data/raw/train_naiverag_squad.csv",
    "../data/raw/train_naiverag_hotpot.csv",
    "../data/raw/train_naiverag_2wiki.csv",
    "../data/raw/train_naiverag_musique.csv",
    "../data/raw/train_lightrag_squad.csv",
    "../data/raw/train_lightrag_hotpot.csv",
    "../data/raw/train_lightrag_2wiki.csv",
    "../data/raw/train_lightrag_musique.csv",
    "../data/raw/train_graphrag_squad.csv",
    "../data/raw/train_graphrag_hotpot.csv",
    "../data/raw/train_graphrag_2wiki.csv",
    "../data/raw/train_graphrag_musique.csv",
]

# 中間パス（特徴量抽出後のデータ, 中間データ）
INTERIM_CSV_FILES: list[str] = [
    "../data/interim/train_naiverag_squad.csv",
    "../data/interim/train_naiverag_hotpot.csv",
    "../data/interim/train_naiverag_2wiki.csv",
    "../data/interim/train_naiverag_musique.csv",
    "../data/interim/train_lightrag_squad.csv",
    "../data/interim/train_lightrag_hotpot.csv",
    "../data/interim/train_lightrag_2wiki.csv",
    "../data/interim/train_lightrag_musique.csv",
    "../data/interim/train_graphrag_squad.csv",
    "../data/interim/train_graphrag_hotpot.csv",
    "../data/interim/train_graphrag_2wiki.csv",
    "../data/interim/train_graphrag_musique.csv",
]

# 中間データの統合用
INTERIM_ALL_CSV_FILES = "../data/interim/train_all.csv"

# 出力パス（訓練データ）
OUTPUT_DIR_TRAIN = "../data/train"
OUTPUT_CSV_FILES_RANK_ALL = f"{OUTPUT_DIR_TRAIN}/train_all_ranks.csv"
OUTPUT_CSV_FILES_RANK_SQUAD = f"{OUTPUT_DIR_TRAIN}/train_squad_ranks.csv"
OUTPUT_CSV_FILES_RANK_HOTPOT = f"{OUTPUT_DIR_TRAIN}/train_hotpot_ranks.csv"
OUTPUT_CSV_FILES_RANK_2WIKI = f"{OUTPUT_DIR_TRAIN}/train_2wiki_ranks.csv"
OUTPUT_CSV_FILES_RANK_MUSIQUE = f"{OUTPUT_DIR_TRAIN}/train_musique_ranks.csv"

# 質問の先頭語
WH_WORDS = ["what", "who", "when", "where", "why", "how", "whose", "which"]
EXTRACTION_WORDS = ["what", "who", "when", "where", "which"]
JUDGMENT_STARTS = [
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "has",
    "have",
    "had",
    "shall",
    "should",
    "may",
    "might",
    "must",
]

# データの列名
BASE_GROUP_COLS: list[str] = [
    "id",
    "question",
    "answer",
    "true_answer",
    "rag_name",
    "model_name",
    "embedding_model_name",
    "chain_type",
    "chunk_size",
    "chunk_overlap",
    "question_length_token",
    "question_named_entities",
    "question_total_number_of_named_entities",
    "question_depth_of_dependency_structure_tree",
    "question_intent",
    "question_type",
    "document_length_token",
    "document_named_entities",
    "document_total_number_of_named_entities",
    "document_keywords",
    "cosine_similarity_of_question_and_document",
    "cosine_similarity_of_question_and_document_named_entities",
    "cosine_similarity_of_question_and_document_keywords",
    "cosine_similarity_of_question_named_entities_and_document",
    "cosine_similarity_of_question_named_entities_and_document_named_entities",
    "cosine_similarity_of_question_named_entities_and_document_keywords",
]

load_dotenv()

# BERTScore の設定
BERTSCORE_MODEL = os.getenv("BERTSCORE_MODEL", "microsoft/deberta-base-mnli")
BERTSCORE_LANG = os.getenv("BERTSCORE_LANG", "en")
BERTSCORE_RESCALE = os.getenv("BERTSCORE_RESCALE", "true").lower() == "true"
BERTSCORE_IDF = os.getenv("BERTSCORE_IDF", "false").lower() == "true"
BERTSCORE_BATCH_SIZE = int(os.getenv("BERTSCORE_BATCH_SIZE", "16"))  # デフォ拡大
BERTSCORE_NUM_LAYERS = os.getenv("BERTSCORE_NUM_LAYERS")
BERTSCORE_CHUNK_SIZE = int(os.getenv("BERTSCORE_CHUNK_SIZE", "512"))

# Sentence-BERT と word2vec の設定
SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
LOCAL_W2V_PATH = os.getenv("LOCAL_W2V_PATH", "")
W2V_FALLBACK = os.getenv("W2V_FALLBACK", "glove-wiki-gigaword-100")

# その他設定
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
_BERT_SCORER: BERTScorer | None = None
_TOKENIZER: AutoTokenizer | None = None
nlp = spacy.load("en_core_web_sm", disable=[])
YAKE_EXTRACTOR = yake.KeywordExtractor(lan="en", n=3, top=5)


# データセット識別
def infer_dataset_from_path(path: str) -> str:
    p = path.lower()
    if "squad" in p:
        return "squad"
    if "hotpot" in p:
        return "hotpot"
    if "2wiki" in p or "2_wiki" in p or "2-wiki" in p:
        return "2wiki"
    if "musique" in p:
        return "musique"
    return "unknown"


# BERTScoreの初期化
def get_bert_scorer() -> BERTScorer:
    global _BERT_SCORER
    if _BERT_SCORER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = dict(
            model_type=BERTSCORE_MODEL,
            lang=BERTSCORE_LANG,
            rescale_with_baseline=BERTSCORE_RESCALE,
            idf=BERTSCORE_IDF,
            device=device,
            batch_size=BERTSCORE_BATCH_SIZE,
        )
        if BERTSCORE_NUM_LAYERS:
            kwargs["num_layers"] = int(BERTSCORE_NUM_LAYERS)
        _BERT_SCORER = BERTScorer(**kwargs)
    return _BERT_SCORER


# BERTScoreモデルに合わせたトークナイザーの取得
def get_tokenizer() -> AutoTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(BERTSCORE_MODEL, use_fast=True)
    return _TOKENIZER


# [0,1] に正規化
def minmax_good(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    s_min, s_max = s.min(), s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_min == s_max:
        return pd.Series(0.0, index=s.index)
    return 1.0 - (s - s_min) / (s_max - s_min)


# リストをカンマ区切りに変換
def safe_join_list(x: Iterable[str] | str) -> str:
    if isinstance(x, str):
        return x
    try:
        return ",".join(map(str, x))
    except Exception:
        return str(x)


_SBERT: SentenceTransformer | None = None
_W2V: gensim.models.KeyedVectors | None = None


# Sentence-BERT を用いた文埋め込み生成モデルのシングルトン取得
def get_sbert() -> SentenceTransformer:
    global _SBERT
    if _SBERT is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _SBERT = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    return _SBERT


# word2vec の取得
def get_w2v() -> gensim.models.KeyedVectors:
    global _W2V
    if _W2V is not None:
        return _W2V

    if LOCAL_W2V_PATH and os.path.exists(LOCAL_W2V_PATH):
        try:
            _W2V = gensim.models.KeyedVectors.load_word2vec_format(
                LOCAL_W2V_PATH,
                binary=LOCAL_W2V_PATH.endswith(".bin"),
            )
            return _W2V
        except Exception:
            pass

    _W2V = gensim_api.load(W2V_FALLBACK)
    return _W2V


#  L2 正規化
@lru_cache(maxsize=200_000)
def _sbert_vec(text: str) -> tuple[float, ...]:
    t = ("" if pd.isna(text) else str(text)).strip()
    if not t:
        return tuple()
    model = get_sbert()
    v = model.encode([t], normalize_embeddings=True)[0]
    return tuple(np.asarray(v, dtype=np.float32))


# word2vec 空白区切りでトークン化
@lru_cache(maxsize=200_000)
def _w2v_vec(text: str) -> tuple[float, ...]:
    t = ("" if pd.isna(text) else str(text)).lower().strip()
    if not t:
        return tuple()
    tokens = [w for w in t.split() if w.isalpha()]
    if not tokens:
        return tuple()

    kv = get_w2v()
    vecs = []
    for w in tokens:
        if w in kv.key_to_index:
            vecs.append(kv[w])
    if not vecs:
        return tuple()
    v = np.mean(vecs, axis=0)
    n = np.linalg.norm(v)
    if n == 0 or np.isnan(n):
        return tuple()
    return tuple((v / n).astype(np.float32))


# L2正規化済みベクトルの内積
def _cos_from_unit_vecs(v1: Iterable[float] | tuple, v2: Iterable[float] | tuple) -> float:
    if not v1 or not v2:
        return 0.0
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    sim = float(np.dot(a, b))
    if np.isnan(sim):
        return 0.0
    return float(np.clip(sim, -1.0, 1.0))


# ここから特徴量抽出
# 依存構造木の深さ
def hop_count_by_dep_doc(doc: spacy.tokens.Doc) -> int:
    max_depth = 1

    def dfs(token, depth: int = 1) -> None:
        nonlocal max_depth
        for child in token.children:
            dfs(child, depth + 1)
        max_depth = max(max_depth, depth)

    for token in doc:
        if token.dep_ == "ROOT":
            dfs(token)

    return max(1, max_depth)


# 特徴量抽出し，指定CSVに保存
def analyze_and_save(input_csv: str, output_csv: str) -> None:
    usecols = None
    df = pd.read_csv(input_csv, usecols=usecols)

    for col in ["question", "context"]:
        if col not in df.columns:
            raise ValueError(f"インプットCSV（{input_csv}） に列 （{col}） が無い")

    nproc = max(1, min(os.cpu_count() or 1, 16))
    q_docs = list(nlp.pipe(df["question"].astype(str).tolist(), n_process=nproc, batch_size=512))
    c_docs = list(nlp.pipe(df["context"].astype(str).tolist(), n_process=nproc, batch_size=128))

    feats: list[dict[str, object]] = []
    for qdoc, cdoc in zip(q_docs, c_docs, strict=False):
        # 質問側
        q_text = qdoc.text
        lower = q_text.lower().strip()
        first_word = lower.split()[0] if lower else ""

        # 固有表現
        q_named = [ent.text for ent in qdoc.ents]
        # 依存構造木の深さ
        hop = hop_count_by_dep_doc(qdoc)
        # 質問意図
        question_intent = next((wh for wh in WH_WORDS if lower.startswith(wh)), None)

        # 質問タイプ
        if first_word in JUDGMENT_STARTS:
            qtype = "judgment"
        elif " or " in lower:
            qtype = "selection"
        elif first_word in EXTRACTION_WORDS:
            qtype = "extraction"
        else:
            qtype = "generation"

        # 文書側
        # 固有表現
        c_named = [ent.text for ent in cdoc.ents]
        # キーワード
        keywords = [kw for kw, _ in YAKE_EXTRACTOR.extract_keywords(cdoc.text)]

        # 意味的特徴量
        q_text = qdoc.text
        c_text = cdoc.text
        q_named_str = " ".join(q_named) if q_named else ""
        c_named_str = " ".join(c_named) if c_named else ""
        kw_str = " ".join(keywords) if keywords else ""

        # Sentence-BERT
        s_q = _sbert_vec(q_text)
        s_ctx = _sbert_vec(c_text)
        s_qne = _sbert_vec(q_named_str)
        s_ctx_ne = _sbert_vec(c_named_str)
        s_ctx_kw = _sbert_vec(kw_str)

        # 質問と文書
        cos_q_ctx = _cos_from_unit_vecs(s_q, s_ctx)
        # 質問と文書固有表現
        cos_q_ctx_ne = _cos_from_unit_vecs(s_q, s_ctx_ne)
        # 質問と文書キーワード
        cos_q_ctx_kw = _cos_from_unit_vecs(s_q, s_ctx_kw)
        # 質問固有表現と文書
        cos_qne_ctx = _cos_from_unit_vecs(s_qne, s_ctx)

        # word2vec
        w_qne = _w2v_vec(q_named_str)
        w_ctx_ne = _w2v_vec(c_named_str)
        w_ctx_kw = _w2v_vec(kw_str)

        # 質問固有表現と文書固有表現
        cos_qne_ctx_ne = _cos_from_unit_vecs(w_qne, w_ctx_ne)
        # 質問固有表現と文書キーワード
        cos_qne_ctx_kw = _cos_from_unit_vecs(w_qne, w_ctx_kw)

        feats.append(
            {
                "question_length_token": len(qdoc),
                "question_named_entities": q_named,
                "question_total_number_of_named_entities": len(q_named),
                "question_depth_of_dependency_structure_tree": hop,
                "question_intent": question_intent,
                "question_type": qtype,
                "document_length_token": len(cdoc),
                "document_named_entities": c_named,
                "document_total_number_of_named_entities": len(c_named),
                "document_keywords": ", ".join(keywords),
                "cosine_similarity_of_question_and_document": cos_q_ctx,
                "cosine_similarity_of_question_and_document_named_entities": cos_q_ctx_ne,
                "cosine_similarity_of_question_and_document_keywords": cos_q_ctx_kw,
                "cosine_similarity_of_question_named_entities_and_document": cos_qne_ctx,
                "cosine_similarity_of_question_named_entities_and_document_named_entities": cos_qne_ctx_ne,
                "cosine_similarity_of_question_named_entities_and_document_keywords": cos_qne_ctx_kw,
            },
        )

    features_df = pd.DataFrame(feats)
    features_df["question_named_entities"] = features_df["question_named_entities"].apply(safe_join_list)
    features_df["document_named_entities"] = features_df["document_named_entities"].apply(safe_join_list)

    merged = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"中間データ（{output_csv}）に結果を保存")


# テキストのトークン長を算出する補助関数
def _token_len(text: str) -> int:
    tok = get_tokenizer()
    if not text:
        return 0
    return len(tok(text, add_special_tokens=False)["input_ids"])


# トークン ID 列を固定長チャンクに分割する処理
def _split_ids(ids: list[int], chunk_size: int) -> list[list[int]]:
    return [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]


# 長文テキストをトークン単位で分割し文字列チャンクに変換
def _chunk_text(text: str, chunk_size: int = 512) -> list[str]:
    tok = get_tokenizer()
    if not text:
        return [""]
    enc = tok(text, add_special_tokens=False)
    ids = enc.get("input_ids", [])
    if not ids:
        return [""]
    return [tok.decode(chunk, clean_up_tokenization_spaces=True) for chunk in _split_ids(ids, chunk_size)]


# BERTScoreの出力を[0, 1] に正規化
def _to_0_1_array(arr: np.ndarray) -> np.ndarray:
    if BERTSCORE_RESCALE:
        return np.clip(arr, 0.0, 1.0)
    return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)


# 適合率・再現率・F1 を計算
@lru_cache(maxsize=200_000)
def _score_single_pair(ref: str, cand: str) -> tuple[float, float, float]:
    ref = "" if pd.isna(ref) else str(ref)
    cand = "" if pd.isna(cand) else str(cand)
    if not ref and not cand:
        return 0.0, 0.0, 0.0

    scorer = get_bert_scorer()
    tok = get_tokenizer()

    r_len = _token_len(ref)
    c_len = _token_len(cand)

    # 短文
    if max(r_len, c_len) <= BERTSCORE_CHUNK_SIZE:
        with torch.inference_mode():
            P, R, F = scorer.score([cand], [ref])
        p = float(P.mean().item())
        r = float(R.mean().item())
        f = float(F.mean().item())
        if BERTSCORE_RESCALE:
            p, r, f = (np.clip(p, 0.0, 1.0), np.clip(r, 0.0, 1.0), np.clip(f, 0.0, 1.0))
        else:
            p, r, f = ((p + 1) / 2, (r + 1) / 2, (f + 1) / 2)
            p, r, f = (np.clip(p, 0.0, 1.0), np.clip(r, 0.0, 1.0), np.clip(f, 0.0, 1.0))
        return float(p), float(r), float(f)

    # 長文
    c_chunks = _chunk_text(cand, BERTSCORE_CHUNK_SIZE)
    r_chunks = _chunk_text(ref, BERTSCORE_CHUNK_SIZE)
    m = max(len(c_chunks), len(r_chunks))
    c_chunks += [""] * (m - len(c_chunks))
    r_chunks += [""] * (m - len(r_chunks))

    with torch.inference_mode():
        P, R, F = scorer.score(c_chunks, r_chunks)

    w = []
    for cc, rc in zip(c_chunks, r_chunks, strict=False):
        w.append(
            (len(tok(cc, add_special_tokens=False)["input_ids"]) + len(tok(rc, add_special_tokens=False)["input_ids"]))
            / 2.0,
        )
    w = np.asarray(w, dtype=np.float64)
    wsum = w.sum() if w.sum() > 0 else len(w)

    Pn = _to_0_1_array(P.detach().cpu().numpy().astype(np.float64))
    Rn = _to_0_1_array(R.detach().cpu().numpy().astype(np.float64))
    Fn = _to_0_1_array(F.detach().cpu().numpy().astype(np.float64))

    p = float((Pn * w).sum() / wsum)
    r = float((Rn * w).sum() / wsum)
    f = float((Fn * w).sum() / wsum)
    return p, r, f


# 精度指標（P, R, F1）とコスト指標（time, token）をグループ単位で集計して正規化
def build_group_results(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["answer", "true_answer", "time"]:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' がない")

    has_token_cols = all(c in df.columns for c in ["input_embedding", "input_chat", "output_chat"])
    uniq_pairs = df[["true_answer", "answer"]].astype(str).drop_duplicates().reset_index(drop=True)

    P_list, R_list, F_list = [], [], []
    for r, c in zip(uniq_pairs["true_answer"].tolist(), uniq_pairs["answer"].tolist(), strict=False):
        p, r_, f = _score_single_pair(r, c)
        P_list.append(p)
        R_list.append(r_)
        F_list.append(f)

    uniq_pairs["__P__"] = P_list
    uniq_pairs["__R__"] = R_list
    uniq_pairs["__F__"] = F_list

    df2 = df.merge(uniq_pairs, on=["true_answer", "answer"], how="left")

    if has_token_cols:
        df2 = df2.assign(
            token=(
                df2["input_embedding"].astype(float)
                + df2["input_chat"].astype(float)
                + df2["output_chat"].astype(float)
            ),
        )

    group_cols = BASE_GROUP_COLS.copy()
    if "dataset" in df2.columns and "dataset" not in group_cols:
        group_cols = ["dataset"] + group_cols

    agg = {
        "__P__": "mean",
        "__R__": "mean",
        "__F__": "mean",
        "time": "mean",
    }
    if has_token_cols:
        agg["token"] = "mean"

    res = (
        df2.groupby(group_cols, dropna=False)
        .agg(agg)
        .reset_index()
        .rename(columns={"__P__": "precision", "__R__": "recall", "__F__": "f1"})
    )

    res["time_normalize"] = minmax_good(res["time"])
    res["token_normalize"] = minmax_good(res["token"]) if "token" in res.columns else 0.0

    return res


# 指標の順位を付与
def add_ranks_and_save(df: pd.DataFrame, out_file: str) -> None:
    for col in ["precision", "recall", "f1", "time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 質問単位でランキング
    rank_prec = df.groupby("question")["precision"].rank(ascending=False, method="dense")
    rank_rec = df.groupby("question")["recall"].rank(ascending=False, method="dense")
    rank_f1 = df.groupby("question")["f1"].rank(ascending=False, method="dense")
    rank_time = df.groupby("question")["time"].rank(ascending=True, method="dense")

    df = df.copy()
    df["rank_precision"] = rank_prec.round(0).astype("Int64")
    df["rank_recall"] = rank_rec.round(0).astype("Int64")
    df["rank_f1"] = rank_f1.round(0).astype("Int64")
    df["rank_time"] = rank_time.round(0).astype("Int64")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"出力パス（{out_file}）に保存")


def main() -> None:
    # 1．特徴量抽出（flagで制御）
    if flag:
        for input_csv, output_csv in zip(INPUT_CSV_FILES, INTERIM_CSV_FILES, strict=False):
            analyze_and_save(input_csv, output_csv)

    # 2．全出力CSVを結合
    interim_dfs: list[pd.DataFrame] = []
    for path in INTERIM_CSV_FILES:
        df_i = pd.read_csv(path)
        df_i = df_i.copy()
        df_i["dataset"] = infer_dataset_from_path(path)
        interim_dfs.append(df_i)

    merged_df = pd.concat(interim_dfs, ignore_index=True)
    print(f"評価値算出前：{merged_df.shape}")

    # 3．構成ごとに集計
    res_df = build_group_results(merged_df)
    print(f"評価値算出後：{res_df.shape}")

    # 4．中間データを保存
    os.makedirs(os.path.dirname(INTERIM_ALL_CSV_FILES), exist_ok=True)
    res_df.to_csv(INTERIM_ALL_CSV_FILES, index=False)
    print(f"中間データ（{INTERIM_ALL_CSV_FILES}）に保存")

    # 5．ランキング付与
    add_ranks_and_save(res_df, OUTPUT_CSV_FILES_RANK_ALL)

    # 6．データセット別にフィルタしてランキングCSVを出力
    for ds, outp in [
        ("squad", OUTPUT_CSV_FILES_RANK_SQUAD),
        ("hotpot", OUTPUT_CSV_FILES_RANK_HOTPOT),
        ("2wiki", OUTPUT_CSV_FILES_RANK_2WIKI),
        ("musique", OUTPUT_CSV_FILES_RANK_MUSIQUE),
    ]:
        df_ds = res_df[res_df.get("dataset", "") == ds].copy()
        if len(df_ds) == 0:
            print(f"データ{ds}は無い．{outp}はスキップ.")
            continue
        add_ranks_and_save(df_ds, outp)


if __name__ == "__main__":
    main()
