from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts"
OUT_DIR = ARTIFACT_DIR / "results" / "adaptive_rag_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "test" / "test_all_ranks.csv"
DEFAULT_OUTPUT_CSV = OUT_DIR / "adaptive_rag_baseline_selected_rows.csv"
DEFAULT_SUMMARY_JSON = OUT_DIR / "adaptive_rag_baseline_summary.json"

DEFAULT_CLASSIFIER_MODEL = "starsuzi/Adaptive-RAG-classifier"

QUESTION_ALIASES = ["question", "query", "q"]
RAG_ALIASES = ["rag_name", "rag", "method", "system"]
CHAIN_ALIASES = ["chain_type", "chain", "prompt_type"]
CHUNK_ALIASES = ["chunk_size", "cs", "chunk"]
OVERLAP_ALIASES = ["chunk_overlap", "co", "overlap"]
TIME_ALIASES = ["time", "runtime", "latency"]
PRECISION_ALIASES = ["precision", "pre", "p"]
RECALL_ALIASES = ["recall", "rec", "r"]
F1_ALIASES = ["f1", "f", "f_score", "f-score"]


def _find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    norm = lambda s: re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")
    lookup = {norm(c): c for c in df.columns}
    for alias in names:
        key = norm(alias)
        if key in lookup:
            return lookup[key]
        for cand_key, orig in lookup.items():
            if key == cand_key or key in cand_key:
                return orig
    return None


def _require_col(df: pd.DataFrame, names: list[str], label: str) -> str:
    col = _find_col(df, names)
    if col is None:
        raise ValueError(f"{label} 列が見つかりません: {names}")
    return col


def _first_nonempty(series: pd.Series) -> str:
    values = series.astype(str).fillna("").tolist()
    for value in values:
        value = value.strip()
        if value:
            return value
    return ""


def _predict_complexity_rule_based(questions: list[str]) -> list[int]:
    multistep_patterns = [
        r"\bcompare\b",
        r"\bdifference\b",
        r"\bbetween\b",
        r"\bboth\b",
        r"\brelation\b",
        r"\brelationship\b",
        r"\bwhy\b",
        r"\bhow\b",
        r"\bbefore\b",
        r"\bafter\b",
        r"\band\b",
        r"\bor\b",
        r"\bsummary\b",
        r"\bsummarize\b",
        r"\bexplain\b",
        r"\bwhich of\b",
    ]
    low_cost_patterns = [
        r"\bwho is\b",
        r"\bwhat is\b",
        r"\bwhen was\b",
        r"\bwhere is\b",
        r"\bwho was\b",
        r"\bwhat year\b",
        r"\bwhat city\b",
    ]

    labels: list[int] = []
    for raw_q in questions:
        q = str(raw_q).strip().lower()
        tokens = re.findall(r"\w+", q)

        if any(re.search(p, q) for p in multistep_patterns) and len(tokens) >= 8:
            labels.append(2)
            continue

        if len(tokens) <= 8 and not any(re.search(p, q) for p in multistep_patterns):
            labels.append(0)
            continue

        if any(re.search(p, q) for p in low_cost_patterns):
            labels.append(0)
            continue

        labels.append(1)
    return labels


def _load_classifier(model_name_or_path: str | None):
    if not model_name_or_path:
        return None
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    except Exception:
        return None
    if torch is not None:
        model.eval()
    return tokenizer, model


def _predict_complexity(questions: list[str], model_name_or_path: str | None) -> tuple[list[int], str]:
    classifier = _load_classifier(model_name_or_path)
    if classifier is None:
        return _predict_complexity_rule_based(questions), "rule_based"

    tokenizer, model = classifier
    if torch is None:
        return _predict_complexity_rule_based(questions), "rule_based"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    batch_size = 32
    preds: list[int] = []
    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            batch = questions[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            batch_preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            preds.extend(int(p) for p in batch_preds)

    return preds, "transformers"


def _score_distance(row: pd.Series, target_chunk: int | None, target_overlap: int | None) -> float:
    distance = 0.0
    if target_chunk is not None and pd.notna(row.get("chunk_size", np.nan)):
        distance += abs(float(row["chunk_size"]) - float(target_chunk))
    if target_overlap is not None and pd.notna(row.get("chunk_overlap", np.nan)):
        distance += abs(float(row["chunk_overlap"]) - float(target_overlap))
    return distance


def _pick_by_mask(g: pd.DataFrame, mask: pd.Series, target_chunk: int | None, target_overlap: int | None) -> pd.Series:
    candidates = g.loc[mask].copy()
    if candidates.empty:
        raise ValueError("候補が空です")

    if "chunk_size" in candidates.columns or "chunk_overlap" in candidates.columns:
        candidates["_distance"] = candidates.apply(
            lambda row: _score_distance(row, target_chunk, target_overlap),
            axis=1,
        )
        sort_cols = ["_distance"]
        ascending = [True]
        if "time" in candidates.columns:
            sort_cols.append("time")
            ascending.append(True)
        chosen = candidates.sort_values(sort_cols, ascending=ascending).iloc[0]
        return chosen.drop(labels=["_distance"])

    if "time" in candidates.columns:
        return candidates.sort_values(["time"], ascending=[True]).iloc[0]
    return candidates.iloc[0]


def _select_row_for_complexity(g: pd.DataFrame, complexity: int) -> tuple[pd.Series, str]:
    rag_col = _find_col(g, RAG_ALIASES)
    chain_col = _find_col(g, CHAIN_ALIASES)
    chunk_col = _find_col(g, CHUNK_ALIASES)
    overlap_col = _find_col(g, OVERLAP_ALIASES)

    if complexity == 0:
        if "time" in g.columns:
            return g.sort_values(["time"], ascending=[True]).iloc[0], "min_time"
        return g.iloc[0], "first_row"

    if complexity == 1:
        mask = pd.Series(True, index=g.index)
        if rag_col is not None:
            mask &= g[rag_col].astype(str).str.lower().eq("naiverag")
        if chain_col is not None:
            mask &= g[chain_col].astype(str).str.lower().eq("stuff")
        if mask.any():
            target_chunk = 500 if chunk_col is not None else None
            target_overlap = 50 if overlap_col is not None else None
            return _pick_by_mask(g, mask, target_chunk, target_overlap), "naiverag_stuff"

        if rag_col is not None:
            fallback = g[rag_col].astype(str).str.lower().eq("naiverag")
            if fallback.any():
                return _pick_by_mask(g, fallback, 500 if chunk_col is not None else None, 50 if overlap_col is not None else None), "naiverag_any"

        if "time" in g.columns:
            return g.sort_values(["time"], ascending=[True]).iloc[0], "fallback_min_time"
        return g.iloc[0], "fallback_first_row"

    mask = pd.Series(True, index=g.index)
    if rag_col is not None:
        mask &= g[rag_col].astype(str).str.lower().eq("graphrag")
    if chain_col is not None:
        chain = g[chain_col].astype(str).str.lower()
        local_mask = chain.eq("local") | chain.eq("global")
        if local_mask.any():
            mask &= local_mask
    if mask.any():
        target_chunk = 500 if chunk_col is not None else None
        target_overlap = 100 if overlap_col is not None else None
        return _pick_by_mask(g, mask, target_chunk, target_overlap), "graphrag_local_or_global"

    if rag_col is not None:
        fallback = g[rag_col].astype(str).str.lower().eq("graphrag")
        if fallback.any():
            return _pick_by_mask(g, fallback, 500 if chunk_col is not None else None, 100 if overlap_col is not None else None), "graphrag_any"

    if "time" in g.columns:
        return g.sort_values(["time"], ascending=[True]).iloc[0], "fallback_min_time"
    return g.iloc[0], "fallback_first_row"


def _mean_metrics(df: pd.DataFrame) -> OrderedDict[str, float]:
    required_cols = {"question", "precision", "recall", "f1", "time"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"必要な列が不足しています: {sorted(missing)}")

    grouped = df.groupby("question", dropna=False)
    mean_dict = OrderedDict(
        [
            ("precision", float(grouped["precision"].mean().mean()) * 100.0),
            ("recall", float(grouped["recall"].mean().mean()) * 100.0),
            ("f1", float(grouped["f1"].mean().mean()) * 100.0),
            ("time", float(grouped["time"].mean().mean())),
        ],
    )
    return mean_dict


def run_adaptive_rag_baseline(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    summary_json: Path = DEFAULT_SUMMARY_JSON,
    classifier_model: str | None = DEFAULT_CLASSIFIER_MODEL,
) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)
    question_col = _require_col(df, QUESTION_ALIASES, "question")
    df[question_col] = df[question_col].astype(str).fillna("").str.strip()

    questions = df[question_col].drop_duplicates().tolist()
    complexities, classifier_source = _predict_complexity(questions, classifier_model)
    q_to_complexity = dict(zip(questions, complexities, strict=True))

    selected_rows: list[pd.Series] = []
    for q, g in df.groupby(question_col, sort=False):
        q = str(q)
        complexity = int(q_to_complexity.get(q, 1))
        chosen, selection_reason = _select_row_for_complexity(g.copy(), complexity)
        chosen = chosen.copy()
        chosen["adaptive_complexity_pred"] = complexity
        chosen["adaptive_selection_reason"] = selection_reason
        chosen["adaptive_classifier_source"] = classifier_source
        selected_rows.append(chosen)

    result_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding="utf-8")

    summary = _mean_metrics(result_df)
    summary["num_questions"] = int(result_df[question_col].nunique())
    summary["num_rows"] = int(len(result_df))
    summary["classifier_source"] = classifier_source
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"保存: {output_csv}")
    print(f"保存: {summary_json}")
    return result_df


def main() -> None:
    run_adaptive_rag_baseline()


if __name__ == "__main__":
    main()
