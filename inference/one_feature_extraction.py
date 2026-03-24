from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import gensim
import gensim.downloader as gensim_api
import numpy as np
import pandas as pd
import spacy
import torch
import yake
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

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


# 設定
load_dotenv()

SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
LOCAL_W2V_PATH = os.getenv("LOCAL_W2V_PATH", "")
W2V_FALLBACK = os.getenv("W2V_FALLBACK", "glove-wiki-gigaword-100")

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

_NLP: spacy.language.Language | None = None
_YAKE: yake.KeywordExtractor | None = None
_SBERT: SentenceTransformer | None = None
_W2V: gensim.models.KeyedVectors | None = None


def get_nlp() -> spacy.language.Language:
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=[])
    return _NLP


def get_yake() -> yake.KeywordExtractor:
    global _YAKE
    if _YAKE is None:
        _YAKE = yake.KeywordExtractor(lan="en", n=3, top=5)
    return _YAKE


def get_sbert() -> SentenceTransformer:
    global _SBERT
    if _SBERT is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _SBERT = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    return _SBERT


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


def safe_join_list(x: Iterable[str] | str) -> str:
    if isinstance(x, str):
        return x
    try:
        return ",".join(map(str, x))
    except Exception:
        return str(x)


@lru_cache(maxsize=200_000)
def _sbert_vec(text: str) -> tuple[float, ...]:
    t = ("" if pd.isna(text) else str(text)).strip()
    if not t:
        return tuple()
    model = get_sbert()
    v = model.encode([t], normalize_embeddings=True)[0]
    return tuple(np.asarray(v, dtype=np.float32))


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


# 特徴量抽出
def extract_features_pair(question: str, context: str) -> dict[str, Any]:
    nlp = get_nlp()
    yake_extractor = get_yake()

    qdoc = nlp(str(question))
    cdoc = nlp(str(context))

    q_text = qdoc.text
    lower = q_text.lower().strip()
    first_word = lower.split()[0] if lower else ""

    q_named = [ent.text for ent in qdoc.ents]
    hop = hop_count_by_dep_doc(qdoc)
    question_intent = next((wh for wh in WH_WORDS if lower.startswith(wh)), None)

    if first_word in JUDGMENT_STARTS:
        qtype = "judgment"
    elif " or " in lower:
        qtype = "selection"
    elif first_word in EXTRACTION_WORDS:
        qtype = "extraction"
    else:
        qtype = "generation"

    c_named = [ent.text for ent in cdoc.ents]
    keywords = [kw for kw, _ in yake_extractor.extract_keywords(cdoc.text)]

    c_text = cdoc.text
    q_named_str = " ".join(q_named) if q_named else ""
    c_named_str = " ".join(c_named) if c_named else ""
    kw_str = " ".join(keywords) if keywords else ""

    s_q = _sbert_vec(q_text)
    s_ctx = _sbert_vec(c_text)
    s_qne = _sbert_vec(q_named_str)
    s_ctx_ne = _sbert_vec(c_named_str)
    s_ctx_kw = _sbert_vec(kw_str)

    cos_q_ctx = _cos_from_unit_vecs(s_q, s_ctx)
    cos_q_ctx_ne = _cos_from_unit_vecs(s_q, s_ctx_ne)
    cos_q_ctx_kw = _cos_from_unit_vecs(s_q, s_ctx_kw)
    cos_qne_ctx = _cos_from_unit_vecs(s_qne, s_ctx)

    w_qne = _w2v_vec(q_named_str)
    w_ctx_ne = _w2v_vec(c_named_str)
    w_ctx_kw = _w2v_vec(kw_str)

    cos_qne_ctx_ne = _cos_from_unit_vecs(w_qne, w_ctx_ne)
    cos_qne_ctx_kw = _cos_from_unit_vecs(w_qne, w_ctx_kw)

    return {
        "question_length_token": len(qdoc),
        # "question_named_entities": safe_join_list(q_named),
        # "question_total_number_of_named_entities": len(q_named),
        "question_depth_of_dependency_structure_tree": int(hop),
        # "question_intent": question_intent,
        "question_type": qtype,
        "document_length_token": len(cdoc),
        # "document_named_entities": safe_join_list(c_named),
        # "document_total_number_of_named_entities": len(c_named),
        # "document_keywords": ", ".join(keywords),
        "cosine_similarity_of_question_and_document": float(cos_q_ctx),
        # "cosine_similarity_of_question_and_document_named_entities": float(cos_q_ctx_ne),
        "cosine_similarity_of_question_and_document_keywords": float(cos_q_ctx_kw),
        # "cosine_similarity_of_question_named_entities_and_document": float(cos_qne_ctx),
        "cosine_similarity_of_question_named_entities_and_document_named_entities": float(cos_qne_ctx_ne),
        # "cosine_similarity_of_question_named_entities_and_document_keywords": float(cos_qne_ctx_kw),
        # "question_named_entities_list": q_named,
        # "document_named_entities_list": c_named,
        # "document_keywords_list": keywords,
    }


def output_and_save_json(obj: dict[str, Any], out_json_path: str, pretty: bool = True) -> None:
    if pretty:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(obj, ensure_ascii=False))

    # save
    p = Path(out_json_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2 if pretty else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--features_json")
    args = parser.parse_args()

    feats = extract_features_pair(question=args.question, context=args.context)
    output_and_save_json(feats, out_json_path=args.features_json, pretty=True)


if __name__ == "__main__":
    main()
