from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import ConstantInputWarning
except Exception:

    class ConstantInputWarning(Warning): ...


import joblib

# パス設定
ARTIFACT_DIR = Path("../artifacts").resolve()
REGISTRY_PATH = (ARTIFACT_DIR / "registry" / "rank_model_registry.json").resolve()
OUT_PRED_DIR = ARTIFACT_DIR / "results" / "selections"
OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)
OUT_METRICS_JSON = (ARTIFACT_DIR / "results" / "test_rank_metrics.json").resolve()

# テストデータ選択
TEST_CSV = Path("../data/test/test_all_ranks.csv").resolve()

# 重みw と 上位k の指定
W_PRE, W_REC, W_F1, W_TIME, W_TOKEN = 0.00, 0.00, 1.00, 0.00, 0.00
ATk = 1

# ランキングモデル
RANK_MODEL_NAME_BY_TASK: dict[str, list[str]] = {
    "precision": ["LightGBM"],
    "recall": ["LightGBM"],
    "f1": ["LightGBM"],
    "time": ["LightGBM"],
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


def _find_col(df: pd.DataFrame, names: Iterable[str]) -> str | None:
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


def _normalize_weights(w):
    s = float(sum(w))
    return [wi / s for wi in w] if s > 0 else [0, 0, 0, 0, 0]


def _pick_default_model(saved_models: dict[str, str], task: str) -> str:
    for name in RANK_MODEL_NAME_BY_TASK.get(task, []):
        if name in saved_models:
            return name
    return next(iter(saved_models.keys()))


# ランキングに基づくRAG構成選択
def run_hyperparam_selection(select_top_k: int) -> None:
    # レジストリ読み込み
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry: dict[str, Any] = json.load(f)
    tasks: list[str] = registry["tasks"]

    task_to_rank_col = {}
    for t in tasks:
        any_model_path = list(registry["saved_models"][t].values())[0]
        bundle = joblib.load(any_model_path)
        task_to_rank_col[t] = bundle.get("rank_label", "")

    # テストデータ読み込み
    df = pd.read_csv(TEST_CSV)
    if "question" not in df.columns:
        raise ValueError("質問がない")
    qid = df["question"].astype(str).fillna("")

    # ランキング予測
    pred_rank_cols: dict[str, str] = {}
    for task in tasks:
        saved_models: dict[str, str] = registry["saved_models"][task]

        # 各モデルで予測して CSV 保存
        for model_name, model_path in saved_models.items():
            bundle = joblib.load(model_path)
            preprocessor = bundle["preprocessor"]
            ranker = bundle["ranker"]
            feat_cols_saved = bundle["feature_cols_in_order"]

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

            # 予測結果の保存
            pred_out = pd.DataFrame({"question": qid, f"pred_{task}_{model_name}": y_pred})
            if "hyperparam_id" in df.columns:
                pred_out["hyperparam_id"] = df["hyperparam_id"]
            pred_path = OUT_PRED_DIR / f"pred_{task}_{model_name}.csv"
            pred_out.to_csv(pred_path, index=False, encoding="utf-8")
            print(f"保存：{pred_path.name}")

        default_model = _pick_default_model(saved_models, task)

        bundle = joblib.load(saved_models[default_model])
        preprocessor = bundle["preprocessor"]
        ranker = bundle["ranker"]
        feat_cols_saved = bundle["feature_cols_in_order"]

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
        "P_rank": W_PRE,
        "R_rank": W_REC,
        "F_rank": W_F1,
        "C_time_rank": W_TIME,
        "C_token_rank": W_TOKEN,
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
        if col is not None:
            sources[key] = col
            used_weights[key] = w

    if not sources:
        raise ValueError("ランクなし")

    ws = np.array([used_weights[k] for k in sources], dtype=float)
    ws = ws / ws.sum()

    inv_terms = []
    for _, col in sources.items():
        inv_terms.append(_safe_inv(df[col]))
    inv_terms = np.vstack(inv_terms)  # shape=(num_terms, N)

    df["Evaluation_score_pred"] = np.dot(ws, inv_terms)

    # 上位kの抽出
    df_sorted = df.sort_values(["question", "Evaluation_score_pred"], ascending=[True, False]).copy()
    topk_df = df_sorted.groupby("question", as_index=False, group_keys=False).head(select_top_k).copy()

    # 保存
    out_csv = OUT_PRED_DIR / "test_all_ranks_scored.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n保存： {out_csv}")

    # 8) best(@1) と top@k の出力（元コード踏襲）
    desired_cols = [
        "rag_name",
        "model_name",
        "embedding_model_name",
        "chain_type",
        "chunk_size",
        "chunk_overlap",
        "time",
        "precision",
        "recall",
        "f1",
    ]
    exist_param_cols = [c for c in desired_cols if c in df.columns]
    missing = [c for c in desired_cols if c not in df.columns]
    if missing:
        print(f"列が見つからない：{missing}")

    best_idx = df.groupby(qid)["Evaluation_score_pred"].idxmax()
    best_rows = df.loc[best_idx, ["question"] + exist_param_cols].sort_values(["question"]).reset_index(drop=True)
    out_best_params = OUT_PRED_DIR / "best_hyperparams_by_question_top1.csv"
    best_rows.to_csv(out_best_params, index=False, encoding="utf-8")
    print(f"保存：{out_best_params}")

    if select_top_k > 1:
        topk_cols = ["question"] + exist_param_cols
        out_topk_params = OUT_PRED_DIR / f"best_hyperparams_by_question_top{select_top_k}.csv"
        topk_df[topk_cols].sort_values(["question"]).to_csv(out_topk_params, index=False, encoding="utf-8")
        print(f"保存：上位 {select_top_k} ，{out_topk_params}")


OUTPUT_CSV_FILES: list[str] = [
    "../data/test/test_all_ranks.csv",
]

KEY_COLS = [
    "question",
    "rag_name",
    "model_name",
    "embedding_model_name",
    "chain_type",
    "chunk_size",
    "chunk_overlap",
    "time",
    "precision",
    "recall",
    "f1",
]

ANSWER_ALIASES = ["answer", "pred_answer", "output", "response"]
TRUE_ANS_ALIASES = ["true_answer", "gold_answer", "reference", "ground_truth", "label"]


def _read_best_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in ["question"] + KEY_COLS[1:] if c not in df.columns]
    if missing:
        raise ValueError(missing)

    df["question"] = df["question"].astype(str).str.strip()
    for c in ["rag_name", "model_name", "embedding_model_name", "chain_type"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in ["chunk_size", "chunk_overlap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def _read_all_results(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        pth = Path(p)
        if not pth.exists():
            print(f"結果が見つからない： {pth}")
            continue
        df = pd.read_csv(pth)

        for c in ["question"]:
            if c not in df.columns:
                print(f"{pth.name} に {c} 列がない")

        if "question" in df.columns:
            df["question"] = df["question"].astype(str).str.strip()
        for c in ["rag_name", "model_name", "embedding_model_name", "chain_type"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        for c in ["chunk_size", "chunk_overlap"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

        df["_source_file"] = pth.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError("CSVなし")
    return pd.concat(frames, ignore_index=True)


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# 評価用に正答を追加
def run_add_answers(select_top_k: int) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "test_predictions").mkdir(parents=True, exist_ok=True)

    in_best_path = ARTIFACT_DIR / "results" / "selections" / f"best_hyperparams_by_question_top{select_top_k}.csv"
    out_matched = ARTIFACT_DIR / "results" / "selections" / f"answers_for_best_hyperparams_top{select_top_k}.csv"
    out_unmatched = ARTIFACT_DIR / "results" / "selections" / "answers_for_best_hyperparams_unmatched.csv"

    best_df = _read_best_df(in_best_path)
    all_df = _read_all_results(OUTPUT_CSV_FILES)

    ans_col = _pick_first_existing(all_df, ANSWER_ALIASES)
    true_col = _pick_first_existing(all_df, TRUE_ANS_ALIASES)

    need_cols = [c for c in KEY_COLS if c in all_df.columns] + [ans_col, true_col, "_source_file"]
    all_sub = all_df[need_cols].copy()

    merged = best_df.merge(
        all_sub,
        how="left",
        on=[c for c in KEY_COLS if c in all_sub.columns],
        validate="one_to_one",
        suffixes=("", "_y"),
    )

    matched_mask = merged[ans_col].notna() & merged[true_col].notna()
    matched = merged.loc[matched_mask].copy()
    unmatched = merged.loc[~matched_mask].copy()

    out_cols = ["question"] + KEY_COLS[1:] + [ans_col, true_col]
    out_cols = [c for c in out_cols if c in matched.columns]
    matched_out = matched[out_cols].rename(columns={ans_col: "answer", true_col: "true_answer"})

    matched_out.to_csv(out_matched, index=False, encoding="utf-8")
    print(f"保存：{out_matched} ({len(matched_out)})")

    if len(unmatched) > 0:
        un_cols = ["question"] + KEY_COLS[1:]
        un_cols = [c for c in un_cols if c in unmatched.columns]
        unmatched[un_cols].to_csv(out_unmatched, index=False, encoding="utf-8")
        print(f"一部の行が見つからない {out_unmatched} ({len(unmatched)})")
    else:
        print("すべての行で完了")


# 評価
def run_score_summary(select_top_k: int) -> None:
    in_dir = ARTIFACT_DIR / "results" / "selections"
    in_name = f"answers_for_best_hyperparams_top{select_top_k}.csv"
    out_name = f"mean_values_top{select_top_k}.json"

    in_matched = in_dir / in_name
    out_matched = ARTIFACT_DIR / "results" / out_name

    df = pd.read_csv(in_matched)

    required_cols = {"question", "precision", "recall", "f1", "time", "answer", "true_answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"必要な列が不足: {sorted(missing)}")

    g = df.groupby("question", dropna=False)

    # 評価方法（平均を取る）
    mean_precision = g["precision"].mean().mean()
    mean_recall = g["recall"].mean().mean()
    mean_f1 = g["f1"].mean().mean()
    mean_time = g["time"].mean().mean()

    # %で表示
    mean_dict = OrderedDict(
        [
            ("precision", float(mean_precision) * 100),
            ("recall", float(mean_recall) * 100),
            ("f1", float(mean_f1) * 100),
            ("time", float(mean_time)),
        ],
    )

    with open(out_matched, "w", encoding="utf-8") as f:
        json.dump(mean_dict, f, ensure_ascii=False, indent=4)

    print(
        f"@{select_top_k}を {out_matched} に保存",
    )


def main() -> None:
    k = ATk
    print(f"開始：topk=@{k}")

    run_hyperparam_selection(k)
    run_add_answers(k)
    run_score_summary(k)

    print(f"終了：topk=@{k}")


if __name__ == "__main__":
    main()
