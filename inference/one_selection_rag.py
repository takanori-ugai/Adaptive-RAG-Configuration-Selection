from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# パス設定
ARTIFACT_DIR = Path("../artifacts").resolve()
REGISTRY_PATH = (ARTIFACT_DIR / "registry" / "rank_model_registry.json").resolve()
OUT_PRED_DIR = ARTIFACT_DIR / "results" / "selections"
OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT_JSON = Path(ARTIFACT_DIR / "results" / "one_qd_features.json").resolve()

# モデル
RANK_MODEL_NAME_BY_TASK: dict[str, list[str]] = {
    "precision": ["LightGBM", "XGBoost", "CatBoost"],
    "recall": ["LightGBM", "XGBoost", "CatBoost"],
    "f1": ["LightGBM", "XGBoost", "CatBoost"],
    "time": ["LightGBM", "XGBoost", "CatBoost"],
}

# エイリアス
COL_ALIASES = {
    "P_rank": ["p_rank", "precision_rank", "pre_rank", "rank_p", "p-rank"],
    "R_rank": ["r_rank", "recall_rank", "rank_r", "r-rank"],
    "F_rank": ["f_rank", "f1_rank", "f-score_rank", "rank_f", "f-rank"],
    "C_time_rank": ["c_time_rank", "time_rank", "latency_rank", "runtime_rank", "rank_time", "c-time-rank"],
    "C_token_rank": ["c_token_rank", "rank_token", "token_rank", "tokens_rank", "cost_token_rank", "c-token-rank"],
}

# タスク名
TASK_TO_SCORE_KEY = {
    "precision": "P_rank",
    "recall": "R_rank",
    "f1": "F_rank",
    "time": "C_time_rank",
    "token": "C_token_rank",
}

# RAG構成
CAND_COLS = [
    "rag_name",
    "model_name",
    "embedding_model_name",
    "chain_type",
    "chunk_size",
    "chunk_overlap",
]


def _find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    norm = lambda s: s.lower().replace("-", "_")
    cand_norm = {norm(n): n for n in df.columns}
    for alias in names:
        key = norm(alias)
        if key in cand_norm:
            return cand_norm[key]
        for k, orig in cand_norm.items():
            if key == k or key in k:
                return orig
    return None


def _safe_inv(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / arr
    inv[~np.isfinite(inv)] = 0.0
    inv[arr <= 0] = 0.0
    return inv


def _pick_default_model(saved_models: dict[str, str], task: str) -> str:
    for name in RANK_MODEL_NAME_BY_TASK.get(task, []):
        if name in saved_models:
            return name
    return next(iter(saved_models.keys()))


def _looks_numeric(col: str) -> bool:
    key = col.lower()
    numeric_hints = [
        "length",
        "total_number",
        "depth",
        "cosine",
        "similarity",
        "count",
        "num_",
        "ratio",
        "score",
    ]
    return any(h in key for h in numeric_hints)


def _stringify_value(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (int, float, np.integer, np.floating)):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        try:
            return ",".join(str(x) for x in v)
        except Exception:
            return json.dumps(v, ensure_ascii=False)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _load_one_pair_features(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSONが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("入力JSONは dict 形式を想定")
    return obj


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray, pd.Index)):
        return list(x)
    return [x]


def _extract_categories_from_column_transformer(ct, col_to_cats: dict[str, set]) -> None:
    try:
        transformers = getattr(ct, "transformers_", None)
        if transformers is None:
            return
    except Exception:
        return

    for _, trans, cols in transformers:
        if trans == "drop" or trans is None:
            continue

        cols_list = _as_list(cols)

        enc = trans
        if hasattr(trans, "steps"):
            try:
                enc = trans.steps[-1][1]
            except Exception:
                enc = trans

        cats = getattr(enc, "categories_", None)
        if cats is None:
            continue
        try:
            for i, c in enumerate(cols_list):
                if c in col_to_cats:
                    col_to_cats[c].update(list(cats[i]))
        except Exception:
            continue


def _extract_categories_from_preprocessor(preprocessor, target_cols: list[str]) -> dict[str, set]:
    col_to_cats = {c: set() for c in target_cols}

    if preprocessor.__class__.__name__ == "ColumnTransformer":
        _extract_categories_from_column_transformer(preprocessor, col_to_cats)
        return col_to_cats

    visited = set()
    stack = [preprocessor]

    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in visited:
            continue
        visited.add(oid)

        if obj.__class__.__name__ == "ColumnTransformer" or hasattr(obj, "transformers_"):
            _extract_categories_from_column_transformer(obj, col_to_cats)

        if hasattr(obj, "steps"):
            try:
                for _, step_obj in obj.steps:
                    stack.append(step_obj)
            except Exception:
                pass

        if hasattr(obj, "transformers_"):
            try:
                for _, trans, _ in obj.transformers_:
                    if trans not in (None, "drop"):
                        stack.append(trans)
            except Exception:
                pass

        for attr in ("named_steps", "transformer_list"):
            if hasattr(obj, attr):
                try:
                    v = getattr(obj, attr)
                    if isinstance(v, dict):
                        stack.extend(list(v.values()))
                    elif isinstance(v, list):
                        stack.extend([x[1] if isinstance(x, tuple) and len(x) == 2 else x for x in v])
                except Exception:
                    pass

    return col_to_cats


def _reconstruct_candidate_space_from_registry(registry: dict[str, Any]) -> pd.DataFrame:
    col_to_vals: dict[str, set] = {c: set() for c in CAND_COLS}
    for task, model_map in registry.get("saved_models", {}).items():
        for model_name, model_path in model_map.items():
            try:
                bundle = joblib.load(model_path)
                pre = bundle["preprocessor"]
                cats = _extract_categories_from_preprocessor(pre, CAND_COLS)
                for c in CAND_COLS:
                    col_to_vals[c].update(cats.get(c, set()))
            except Exception:
                continue

    empties = [c for c, s in col_to_vals.items() if len(s) == 0]
    if empties:
        raise ValueError("registry/joblib から候補空間を復元失敗")

    def _normalize_chunk(x: Any) -> Any:
        try:
            if isinstance(x, (np.integer, int)):
                return int(x)
            if isinstance(x, (np.floating, float)) and float(x).is_integer():
                return int(x)
            s = str(x)
            if s.isdigit():
                return int(s)
            f = float(s)
            if f.is_integer():
                return int(f)
        except Exception:
            pass
        return x

    col_to_vals["chunk_size"] = {_normalize_chunk(x) for x in col_to_vals["chunk_size"]}
    col_to_vals["chunk_overlap"] = {_normalize_chunk(x) for x in col_to_vals["chunk_overlap"]}

    grids = [sorted(list(col_to_vals[c]), key=lambda z: str(z)) for c in CAND_COLS]
    rows = list(itertools.product(*grids))
    df = pd.DataFrame(rows, columns=CAND_COLS)

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _build_input_df(
    question: str,
    context: str,
    one_pair_features: dict[str, Any],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    df = candidates.copy()
    df.insert(0, "question", question)
    df["context"] = context

    for k, v in one_pair_features.items():
        if k in ("question", "context"):
            continue
        df[k] = _stringify_value(v)

    if "chunk_size" in df.columns:
        df["chunk_size"] = pd.to_numeric(df["chunk_size"], errors="coerce")
    if "chunk_overlap" in df.columns:
        df["chunk_overlap"] = pd.to_numeric(df["chunk_overlap"], errors="coerce")

    return df


def _ensure_required_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if not missing:
        return df
    for c in missing:
        df[c] = 0.0 if _looks_numeric(c) else ""
    print(f"特徴量列が不足: {missing}")
    return df


def run_hyperparam_selection_from_json(
    *,
    question: str,
    context: str,
    features_json_path: Path,
    select_top_k: int,
    w_pre: float,
    w_rec: float,
    w_f1: float,
    w_time: float,
    w_token: float,
) -> None:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"registry が見つかりません: {REGISTRY_PATH}")

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry: dict[str, Any] = json.load(f)

    tasks: list[str] = registry["tasks"]
    one_pair_features = _load_one_pair_features(features_json_path)

    candidates = _reconstruct_candidate_space_from_registry(registry)
    print(f"[INFO] reconstructed candidates: {len(candidates)} rows")

    df = _build_input_df(question, context, one_pair_features, candidates)
    qid = df["question"].astype(str).fillna("")

    pred_rank_cols: dict[str, str] = {}

    for task in tasks:
        saved_models: dict[str, str] = registry["saved_models"][task]

        for model_name, model_path in saved_models.items():
            bundle = joblib.load(model_path)
            preprocessor = bundle["preprocessor"]
            ranker = bundle["ranker"]
            feat_cols_saved = bundle["feature_cols_in_order"]

            df = _ensure_required_columns(df, feat_cols_saved)
            X_in = df[feat_cols_saved].copy()
            X_t = preprocessor.transform(X_in)

            if hasattr(ranker, "feature_name_") and getattr(ranker, "feature_name_", None) is not None:
                feat_names = list(ranker.feature_name_)
                try:
                    X_for_pred = pd.DataFrame(X_t, columns=feat_names)
                except Exception:
                    X_for_pred = X_t
            else:
                X_for_pred = X_t

            y_pred = np.asarray(ranker.predict(X_for_pred)).ravel()

            pred_out = pd.DataFrame({"question": qid, f"pred_{task}_{model_name}": y_pred})
            pred_path = OUT_PRED_DIR / f"pred_{task}_{model_name}.csv"
            pred_out.to_csv(pred_path, index=False, encoding="utf-8")
            print(f"保存：{pred_path.name}")

        default_model = _pick_default_model(saved_models, task)
        bundle = joblib.load(saved_models[default_model])
        preprocessor = bundle["preprocessor"]
        ranker = bundle["ranker"]
        feat_cols_saved = bundle["feature_cols_in_order"]

        df = _ensure_required_columns(df, feat_cols_saved)
        X_in = df[feat_cols_saved].copy()
        X_t = preprocessor.transform(X_in)

        if hasattr(ranker, "feature_name_") and getattr(ranker, "feature_name_", None) is not None:
            feat_names = list(ranker.feature_name_)
            try:
                X_for_pred = pd.DataFrame(X_t, columns=feat_names)
            except Exception:
                X_for_pred = X_t
        else:
            X_for_pred = X_t

        y_pred = np.asarray(ranker.predict(X_for_pred)).ravel()
        pred_score_col = f"pred_{task}_{default_model}"
        df[pred_score_col] = y_pred

        pred_rank_col = f"{TASK_TO_SCORE_KEY.get(task, task)}_pred_rank"
        df[pred_rank_col] = df.groupby(qid)[pred_score_col].rank(method="first", ascending=False).astype(int)

        score_key = TASK_TO_SCORE_KEY.get(task)
        if score_key:
            pred_rank_cols[score_key] = pred_rank_col

        print(f"タスク：{task}， モデル：{default_model}")

    WEIGHT_MAP = {
        "P_rank": float(w_pre),
        "R_rank": float(w_rec),
        "F_rank": float(w_f1),
        "C_time_rank": float(w_time),
        "C_token_rank": float(w_token),
    }

    def _try_get_rank_source(score_key: str) -> tuple[str | None, float]:
        if score_key in pred_rank_cols:
            return pred_rank_cols[score_key], WEIGHT_MAP[score_key]
        found = _find_col(df, COL_ALIASES[score_key])
        if found is not None and found in df.columns:
            return found, WEIGHT_MAP[score_key]
        return None, 0.0

    sources: dict[str, str] = {}
    used_weights: dict[str, float] = {}

    for key in ["P_rank", "R_rank", "F_rank", "C_time_rank", "C_token_rank"]:
        col, w = _try_get_rank_source(key)
        if col is not None and float(w) != 0.0:
            sources[key] = col
            used_weights[key] = w

    if not sources:
        raise ValueError("重みが0以外のランク列が見つかりません")

    ws = np.array([used_weights[k] for k in sources], dtype=float)
    if float(ws.sum()) <= 0:
        raise ValueError("重みの合計が0")
    ws = ws / ws.sum()

    inv_terms = np.vstack([_safe_inv(df[col]) for col in sources.values()])
    df["Evaluation_score"] = np.dot(ws, inv_terms)

    df_sorted = df.sort_values(["question", "Evaluation_score"], ascending=[True, False]).copy()
    topk_df = df_sorted.groupby("question", as_index=False, group_keys=False).head(select_top_k).copy()

    out_csv = OUT_PRED_DIR / "test_all_ranks_scored.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n保存： {out_csv}")

    best_idx = df.groupby(qid)["Evaluation_score"].idxmax()
    best_rows = df.loc[best_idx, ["question"] + CAND_COLS].sort_values(["question"]).reset_index(drop=True)
    out_best_params = OUT_PRED_DIR / "best_hyperparams_by_question_top1.csv"
    best_rows.to_csv(out_best_params, index=False, encoding="utf-8")
    print(f"保存：{out_best_params}")

    if select_top_k > 1:
        out_topk_params = OUT_PRED_DIR / f"best_hyperparams_by_question_top{select_top_k}.csv"
        topk_df[["question"] + CAND_COLS + ["Evaluation_score"]].to_csv(
            out_topk_params,
            index=False,
            encoding="utf-8",
        )
        print(f"保存：上位 {select_top_k} ，{out_topk_params}")

    print("\n RAG構成")
    print("☆")
    print(topk_df[CAND_COLS + ["Evaluation_score"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--features_json")

    parser.add_argument("--w_pre", type=float, default=0.0)
    parser.add_argument("--w_rec", type=float, default=0.0)
    parser.add_argument("--w_f1", type=float, default=1.0)
    parser.add_argument("--w_time", type=float, default=0.0)
    parser.add_argument("--w_token", type=float, default=0.0)
    parser.add_argument("--atk", type=int, default=1, help="上位k（@k）")

    args = parser.parse_args()

    features_path = Path(args.features_json).resolve() if args.features_json else DEFAULT_INPUT_JSON

    print(f"開始：topk=@{args.atk}")
    print(f"入力JSON：{features_path}")
    print(f"重み: pre={args.w_pre}, rec={args.w_rec}, f1={args.w_f1}, time={args.w_time}, token={args.w_token}")

    run_hyperparam_selection_from_json(
        question=args.question,
        context=args.context,
        features_json_path=features_path,
        select_top_k=args.atk,
        w_pre=args.w_pre,
        w_rec=args.w_rec,
        w_f1=args.w_f1,
        w_time=args.w_time,
        w_token=args.w_token,
    )

    print(f"終了：topk=@{args.atk}")


if __name__ == "__main__":
    main()
