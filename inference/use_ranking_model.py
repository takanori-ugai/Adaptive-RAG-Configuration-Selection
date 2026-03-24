from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import ndcg_score

try:
    from scipy.stats import ConstantInputWarning
except Exception:

    class ConstantInputWarning(Warning): ...


import joblib

# パス設定
ARTIFACT_DIR = Path("../artifacts").resolve()
REGISTRY_PATH = (ARTIFACT_DIR / "registry" / "rank_model_registry.json").resolve()
TEST_CSV = Path("../data/test/test_all_ranks.csv").resolve()
OUT_PRED_DIR = ARTIFACT_DIR / "results" / "predictions"
OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)
OUT_METRICS_JSON = (ARTIFACT_DIR / "results" / "test_rank_metrics.json").resolve()


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        r_full = pearsonr(x, y)
    r = r_full.statistic if hasattr(r_full, "statistic") else r_full[0]
    return float(0.0 if np.isnan(r) else r)


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        r = spearmanr(x, y).correlation
    return float(0.0 if (r is None or np.isnan(r)) else r)


def _kendall_tau_b_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        r = kendalltau(x, y, variant="b").correlation
    return float(0.0 if (r is None or np.isnan(r)) else r)


def _hit_at_k(y_true_gain: np.ndarray, y_pred_score: np.ndarray, k: int | None) -> float:
    order = np.argsort(-y_pred_score)
    top_idx = order if k is None else order[: min(k, len(order))]
    best = np.max(y_true_gain)
    return 1.0 if np.any(y_true_gain[top_idx] == best) else 0.0


def eval_metrics_per_query(df_q: pd.DataFrame, score_col: str, rel_col_as_gain: str) -> dict[str, float | int]:
    ndcg1, ndcg3, ndcg5, ndcg10, ndcgall = [], [], [], [], []
    hit1, hit3, hit5, hit10, hitall = [], [], [], [], []
    kendalls, spearmans, pearsons = [], [], []

    for _, g in df_q.groupby("_qid", sort=False):
        y_true = g[rel_col_as_gain].to_numpy(dtype=float)
        y_pred = g[score_col].to_numpy(dtype=float)
        if y_true.shape[0] < 2:
            continue

        y_true_2d = y_true.reshape(1, -1)
        y_pred_2d = y_pred.reshape(1, -1)
        ndcg1.append(ndcg_score(y_true_2d, y_pred_2d, k=1))
        ndcg3.append(ndcg_score(y_true_2d, y_pred_2d, k=3))
        ndcg5.append(ndcg_score(y_true_2d, y_pred_2d, k=5))
        ndcg10.append(ndcg_score(y_true_2d, y_pred_2d, k=10))
        ndcgall.append(ndcg_score(y_true_2d, y_pred_2d, k=None))

        hit1.append(_hit_at_k(y_true, y_pred, k=1))
        hit3.append(_hit_at_k(y_true, y_pred, k=3))
        hit5.append(_hit_at_k(y_true, y_pred, k=5))
        hit10.append(_hit_at_k(y_true, y_pred, k=10))
        hitall.append(_hit_at_k(y_true, y_pred, k=None))

        kendalls.append(_kendall_tau_b_safe(y_true, y_pred))
        spearmans.append(_spearman_safe(y_true, y_pred))
        pearsons.append(_pearson_safe(y_true, y_pred))

    m = lambda arr: float(np.mean(arr) if arr else 0.0)
    return {
        "NDCG@1": m(ndcg1),
        "NDCG@3": m(ndcg3),
        "NDCG@5": m(ndcg5),
        "NDCG@10": m(ndcg10),
        "NDCG@ALL": m(ndcgall),
        "Hit@1": m(hit1),
        "Hit@3": m(hit3),
        "Hit@5": m(hit5),
        "Hit@10": m(hit10),
        "Hit@ALL": m(hitall),
        "ケンドール": m(kendalls),
        "スピアマン": m(spearmans),
        "ピアソン": m(pearsons),
        "num_queries": int(df_q["_qid"].nunique()),
    }


def ranks_to_relevance_1_to_64(ranks: pd.Series, groups: pd.Series) -> pd.Series:
    r = ranks.astype(float).copy()
    with np.errstate(invalid="ignore"):
        r_min = np.nanmin(r.values)
    if pd.notna(r_min) and r_min <= 0:
        r = r - r_min + 1.0

    rel = np.zeros(len(r), dtype=np.int32)
    gser = groups.reset_index(drop=True)
    r = r.reset_index(drop=True)

    for _, idx in gser.groupby(gser).groups.items():
        idx = list(idx)
        rr = r.iloc[idx]
        rr_filled = rr.fillna(rr.max() + 1.0)
        max_r = rr_filled.max()

        if not np.isfinite(max_r) or max_r <= 1:
            rel_vals = np.full(len(idx), 1, dtype=np.int32)
        else:
            rel_vals = np.floor((rr_filled - 1.0) / (max_r - 1.0) * 63.0 + 1.0).astype(np.int32)
            rel_vals = np.clip(rel_vals, 1, 64)

        rel[idx] = rel_vals

    return pd.Series(rel.astype(np.int32), index=ranks.index)


def rel_to_gain(rel_series: pd.Series | np.ndarray) -> np.ndarray:
    rel_arr = np.asarray(rel_series, dtype=np.int32)
    rel_arr = np.clip(rel_arr, 1, 64)
    gain = 65 - rel_arr
    return gain.astype(np.int32)


def main() -> None:
    # 1) レジストリ読み込み
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry: dict[str, Any] = json.load(f)

    tasks: list[str] = registry["tasks"]
    feature_cols: list[str] = registry["feature_cols_in_order"]

    task_to_rank_col = {}
    for t in tasks:
        any_model_path = list(registry["saved_models"][t].values())[0]
        bundle = joblib.load(any_model_path)
        task_to_rank_col[t] = bundle["rank_label"]

    df = pd.read_csv(TEST_CSV)
    if "question" not in df.columns:
        raise ValueError("test CSV に 'question' 列が必要")

    qid = df["question"].astype(str).fillna("")
    X = df[feature_cols].copy()

    rel_cols = {}
    gain_cols = {}
    for task in tasks:
        rank_col = task_to_rank_col[task]
        if rank_col not in df.columns:
            raise ValueError(f"test CSV に '{rank_col}' 列がない（{task}）")
        rel = ranks_to_relevance_1_to_64(df[rank_col], qid)
        gain = rel_to_gain(rel)
        rel_cols[task] = rel
        gain_cols[task] = gain

    all_metrics: dict[str, dict[str, float | int]] = {}

    for task in tasks:
        all_metrics[task] = {}
        saved_models: dict[str, str] = registry["saved_models"][task]
        for model_name, model_path in saved_models.items():
            bundle = joblib.load(model_path)
            preprocessor = bundle["preprocessor"]
            ranker = bundle["ranker"]
            feat_cols_saved = bundle["feature_cols_in_order"]

            X_in = df[feat_cols_saved].copy()
            X_t = preprocessor.transform(X_in)

            if hasattr(ranker, "feature_name_") and ranker.feature_name_ is not None:
                feat_names = list(ranker.feature_name_)
                try:
                    X_for_pred = pd.DataFrame(X_t, columns=feat_names)
                except Exception:
                    X_for_pred = X_t
            else:
                X_for_pred = X_t

            y_pred = ranker.predict(X_for_pred)
            y_pred = np.asarray(y_pred).ravel()

            df_eval = pd.DataFrame(
                {
                    "_qid": qid.values,
                    "true_gain": gain_cols[task].astype(float),
                    "pred_score": y_pred.astype(float),
                },
            )
            metrics = eval_metrics_per_query(df_eval, score_col="pred_score", rel_col_as_gain="true_gain")

            out_pred_path = OUT_PRED_DIR / f"pred_{task}_{model_name}.csv"
            cols_to_keep = [
                "question",
                "rag_name",
                "model_name",
                "embedding_model_name",
                "chain_type",
                "chunk_size",
                "chunk_overlap",
            ]
            cols_exist = [c for c in cols_to_keep if c in df.columns]
            out_df = pd.concat(
                [
                    df[cols_exist].reset_index(drop=True),
                    pd.DataFrame(
                        {
                            "true_rank_label": task_to_rank_col[task],
                            "true_rel": rel_cols[task].values,
                            "true_gain": gain_cols[task],
                            "pred_score": y_pred,
                        },
                    ),
                ],
                axis=1,
            )
            out_df.to_csv(out_pred_path, index=False, encoding="utf-8")
            print(f"結果：{out_pred_path.name}")

            all_metrics[task][model_name] = metrics

    with open(OUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"結果：{OUT_METRICS_JSON.name}")


if __name__ == "__main__":
    main()
