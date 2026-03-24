import json
import re
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from lightgbm import LGBMRanker
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRanker

# パス設定
ARTIFACT_DIR = Path("../artifacts")
VISUALIZATION_DIR = Path("../artifacts/figures")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


# flag=False なら検証のみ
flag = True
# flag = False

# 訓練データのパス
CSV_PATH = "../data/train/train_all_ranks.csv"
# CSV_PATH = "../data/train/train_squad_ranks.csv"
# CSV_PATH = "../data/train/train_hotpot_ranks.csv"
# CSV_PATH = "../data/train/train_2wiki_ranks.csv"
# CSV_PATH = "../data/train/train_musique_ranks.csv"
# CSV_PATH = "../data/train/train_all_ranks_gpt.csv"


df = pd.read_csv(CSV_PATH)
group_series = df["question"].astype(str).fillna("")

# 特徴量の定義
categorical_cols = [
    "rag_name",
    "model_name",
    "embedding_model_name",
    "chain_type",
    "chunk_size",
    "chunk_overlap",
    # "question_intent",
    "question_type",
]

numeric_cols = [
    "question_length_token",
    # "question_total_number_of_named_entities",
    "question_depth_of_dependency_structure_tree",
    "document_length_token",
    # "document_total_number_of_named_entities",
    "cosine_similarity_of_question_and_document",
    # "cosine_similarity_of_question_and_document_named_entities",
    "cosine_similarity_of_question_and_document_keywords",
    # "cosine_similarity_of_question_named_entities_and_document",
    "cosine_similarity_of_question_named_entities_and_document_named_entities",
    # "cosine_similarity_of_question_named_entities_and_document_keywords",
]

feature_cols = categorical_cols + numeric_cols
numeric_wo_embed = numeric_cols.copy()
X_all = df[feature_cols].copy()


# 日本語フォント設定
def set_japanese_font():
    import matplotlib as mpl
    import matplotlib.font_manager as fm

    candidates = [
        "IPAexGothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "Yu Gothic",
        "Meiryo",
        "Hiragino Sans",
        "TakaoGothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.family"] = name
            break

    mpl.rcParams["axes.unicode_minus"] = False


# 特徴量貢献度のラベル設定
FEATURE_NAME_JP = {
    # categorical
    "rag_name": "RAG手法",
    "model_name": "LLM",
    "embedding_model_name": "埋め込みモデル",
    "chain_type": "Chain種別",
    "chunk_size": "チャンクサイズ",
    "chunk_overlap": "チャンク重複",
    "question_type": "質問タイプ",
    # numeric
    "question_length_token": "質問トークン数",
    "question_depth_of_dependency_structure_tree": "質問の依存木深さ",
    "document_length_token": "文書トークン数",
    "cosine_similarity_of_question_and_document": "質問-文書 類似度",
    "cosine_similarity_of_question_and_document_keywords": "質問-文書(キーワード) 類似度",
    "cosine_similarity_of_question_named_entities_and_document_named_entities": "質問-文書(固有表現) 類似度",
}


# ピアソン
def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    r = pearsonr(x, y).statistic if hasattr(pearsonr(x, y), "statistic") else pearsonr(x, y)[0]
    return float(0.0 if np.isnan(r) else r)


# スピアマン
def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    r = spearmanr(x, y).correlation
    return float(0.0 if (r is None or np.isnan(r)) else r)


# ケンドール
def _kendall_tau_b_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    r = kendalltau(x, y, variant="b").correlation
    return float(0.0 if (r is None or np.isnan(r)) else r)


# Hit@k
def _hit_at_k(y_true_gain: np.ndarray, y_pred_score: np.ndarray, k: int | None) -> float:
    order = np.argsort(-y_pred_score)
    top_idx = order if k is None else order[: min(k, len(order))]
    best = np.max(y_true_gain)
    return 1.0 if np.any(y_true_gain[top_idx] == best) else 0.0


# 評価値算出
def eval_metrics_per_query(df_q: pd.DataFrame, score_col: str, rel_col_as_gain: str) -> dict:
    ndcg1, ndcg3, ndcg5, ndcg10, ndcgall = [], [], [], [], []
    hit1, hit3, hit5, hit10, hitall = [], [], [], [], []
    kendalls, spearmans, pearsons = [], [], []

    for _, g in df_q.groupby("_qid_ordered", sort=False):
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
        "num_queries": int(df_q["_qid_ordered"].nunique()),
    }


# 特徴量貢献度可視化
def plot_and_save_feature_importance(
    model,
    feature_names: list[str],
    model_name: str,
    task_key: str,
    save_dir: Path,
    suffix: str = "",
    *,
    feature_label_map: dict[str, str] | None = None,
    xlabel_jp: str = "重要度（％）",
    ylabel_jp: str = "特徴量",
    title_jp: str | None = None,
    catboost_data: "Pool | None" = None,  # ★追加：CatBoost用
):
    import numpy as np
    import pandas as pd

    try:
        set_japanese_font()
    except Exception:
        pass

    estimator = model
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    elif hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]

    used_feat_names = list(feature_names) if feature_names is not None else None
    importances = None

    # LightGBM
    try:
        import lightgbm as lgb

        if isinstance(estimator, (lgb.LGBMClassifier, lgb.LGBMRegressor, lgb.LGBMRanker)):
            if getattr(estimator, "booster_", None) is not None:
                arr = estimator.booster_.feature_importance(importance_type="gain")
                if arr is not None:
                    importances = np.asarray(arr, dtype=float).reshape(-1)
                    if hasattr(estimator, "feature_name_") and estimator.feature_name_:
                        cand_names = list(estimator.feature_name_)
                        is_auto_cols = all(re.match(r"^Column_\d+$", n) for n in cand_names)
                        if not is_auto_cols or used_feat_names is None or len(used_feat_names) != len(cand_names):
                            used_feat_names = cand_names
    except Exception:
        pass

    # XGBoost
    if importances is None:
        try:
            import xgboost as xgb

            if isinstance(estimator, (xgb.XGBClassifier, xgb.XGBRegressor, xgb.XGBRanker)):
                booster = estimator.get_booster()
                imp_dict = {}
                for itype in ("gain", "weight", "cover", "total_gain", "total_cover"):
                    imp_dict = booster.get_score(importance_type=itype)
                    if imp_dict:
                        break

                if imp_dict is not None:
                    bnames = booster.feature_names
                    n_feats = None
                    if hasattr(estimator, "n_features_in_"):
                        n_feats = int(estimator.n_features_in_)
                    elif feature_names is not None:
                        n_feats = len(feature_names)
                    elif bnames is not None:
                        n_feats = len(bnames)

                    def dict_to_array_by_index(d, size):
                        arr = np.zeros(size, dtype=float)
                        for k, v in d.items():
                            m = re.match(r"^f(\d+)$", k)
                            if m:
                                idx = int(m.group(1))
                                if 0 <= idx < size:
                                    arr[idx] = float(v)
                        return arr

                    if bnames and all(re.match(r"^f\d+$", n) for n in bnames):
                        size = len(bnames) if n_feats is None else n_feats
                        importances = dict_to_array_by_index(imp_dict, size)
                        if feature_names is not None and len(feature_names) >= size:
                            used_feat_names = list(feature_names)[:size]
                        else:
                            used_feat_names = [f"f{i}" for i in range(size)]
                    elif bnames:
                        used_feat_names = list(bnames)
                        importances = np.array([float(imp_dict.get(n, 0.0)) for n in used_feat_names], dtype=float)
                    else:
                        keys = list(imp_dict.keys())
                        if all(re.match(r"^f\d+$", k) for k in keys) and n_feats is not None:
                            importances = dict_to_array_by_index(imp_dict, n_feats)
                            used_feat_names = (
                                list(feature_names)[:n_feats]
                                if feature_names is not None and len(feature_names) >= n_feats
                                else [f"f{i}" for i in range(n_feats)]
                            )
                        else:
                            used_feat_names = keys
                            importances = np.array([float(imp_dict[k]) for k in keys], dtype=float)
        except Exception:
            pass

    # CatBoost
    if importances is None:
        try:
            from catboost import CatBoost

            if isinstance(estimator, CatBoost):
                if catboost_data is not None:
                    arr = estimator.get_feature_importance(
                        data=catboost_data,
                        type="PredictionValuesChange",
                    )
                else:
                    arr = estimator.get_feature_importance(type="PredictionValuesChange")

                if arr is not None:
                    importances = np.asarray(arr, dtype=float).reshape(-1)
        except Exception:
            pass

    if importances is None and hasattr(estimator, "feature_importances_"):
        fi = getattr(estimator, "feature_importances_", None)
        if fi is not None:
            importances = np.asarray(fi, dtype=float).reshape(-1)

    if importances is None or getattr(importances, "ndim", 0) == 0:
        print(f"{model_name}: feature importances を取得できず。プロットをスキップします。")
        return

    if used_feat_names is not None and all(re.match(r"^(f\d+|Column_\d+)$", n) for n in used_feat_names):
        if feature_names is not None and len(feature_names) >= len(importances):
            used_feat_names = list(feature_names)[: len(importances)]

    if used_feat_names is None:
        used_feat_names = [f"f{i}" for i in range(len(importances))]

    n = min(len(importances), len(used_feat_names))
    importances = importances[:n]
    used_feat_names = used_feat_names[:n]
    importances = np.abs(importances)

    # 正規化（％）
    total = float(np.nansum(importances))
    if np.isfinite(total) and total > 0:
        vals = importances / total * 100.0
    else:
        vals = importances

    # 日本語ラベル
    label_map = feature_label_map or {}
    display_names = [label_map.get(f, f) for f in used_feat_names]

    df_importance = (
        pd.DataFrame({"feature": display_names, "importance": vals})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(12, 8))
    y = np.arange(len(df_importance))
    plt.barh(y, df_importance["importance"].to_numpy())
    plt.yticks(y, df_importance["feature"].tolist())
    plt.gca().invert_yaxis()

    plt.xlabel(xlabel_jp)
    plt.ylabel(ylabel_jp)
    if title_jp is None:
        title_jp = f"特徴量貢献度：{model_name}，{task_key}，{suffix}"
    plt.title(title_jp, fontsize=16)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"feature_importance_{task_key}_{model_name}{suffix}.png"
    plt.savefig(save_dir / filename)
    plt.close()
    print(f"特徴量重要度グラフを保存: {filename}")


set_japanese_font()


# ランキング用ラベル
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


rank_cols_map = {
    "precision": "rank_precision",
    "recall": "rank_recall",
    "f1": "rank_f1",
    "time": "rank_time",
}

rel_cols = {}
for key, rank_col in rank_cols_map.items():
    if rank_col not in df.columns:
        raise ValueError(f"CSVに {rank_col} 列が見つからない")
    rel_cols[key] = ranks_to_relevance_1_to_64(df[rank_col], group_series)

# 前処理
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
        ("num", StandardScaler(), numeric_wo_embed),
    ],
    remainder="drop",
)

# ホールドアウト検証
print("【ホールドアウト検証】")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_all, groups=group_series))

X_train = X_all.iloc[train_idx].copy()
X_test = X_all.iloc[test_idx].copy()
qid_train_raw = group_series.iloc[train_idx].reset_index(drop=True)
qid_test_raw = group_series.iloc[test_idx].reset_index(drop=True)

preprocessor.fit(X_train)
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)


# 学習用に並び替え
def order_by_query(X_t: np.ndarray, qids: pd.Series, rel_dict: dict[str, pd.Series], original_index: pd.Index):
    tmp = pd.DataFrame({"qid": qids, "_order": np.arange(len(qids))}, index=pd.Index(range(len(qids))))
    tmp_sorted = tmp.sort_values(["qid", "_order"], kind="mergesort").reset_index(drop=True)
    ordered_idx = tmp_sorted["_order"].to_numpy()
    X_t_ordered = X_t[ordered_idx, :]
    qids_ordered = tmp_sorted["qid"].reset_index(drop=True)

    rel_ordered = {}
    for k, s in rel_dict.items():
        s_part = s.loc[original_index].reset_index(drop=True)
        rel_ordered[k] = s_part.iloc[ordered_idx].reset_index(drop=True)

    return X_t_ordered, qids_ordered, rel_ordered, ordered_idx


rel_train_dict_raw = {k: v.iloc[train_idx] for k, v in rel_cols.items()}
X_train_ord, qid_train_ord, rel_train_dict, _ = order_by_query(
    X_train_t,
    qid_train_raw,
    rel_train_dict_raw,
    original_index=X_all.iloc[train_idx].index,
)

rel_test_dict_raw = {k: v.iloc[test_idx] for k, v in rel_cols.items()}
X_test_ord, qid_test_ord, rel_test_dict, test_order = order_by_query(
    X_test_t,
    qid_test_raw,
    rel_test_dict_raw,
    original_index=X_all.iloc[test_idx].index,
)


def make_group_sizes(qids: pd.Series) -> list[int]:
    return qids.value_counts(sort=False)[qids.unique()].tolist()


group_train = make_group_sizes(qid_train_ord)
group_test = make_group_sizes(qid_test_ord)


def rel_to_gain(rel_series: pd.Series | np.ndarray) -> np.ndarray:
    rel_arr = np.asarray(rel_series, dtype=np.int32)
    rel_arr = np.clip(rel_arr, 1, 64)
    gain = 65 - rel_arr
    return gain.astype(np.int32)


metrics_all = {}
pred_store = {}

# 目的変数ごとに学習
for task_key in ["precision", "recall", "f1", "time"]:
    print(f"\n目的変数：{task_key}")

    y_train_gain = rel_to_gain(rel_train_dict[task_key])
    y_test_gain = rel_to_gain(rel_test_dict[task_key])

    max_gain = int(max(y_train_gain.max(), y_test_gain.max(), 64))
    lgb_gain = list(range(max_gain + 1))

    results_task = {}
    preds_task = {}

    # LightGBM
    lgb_ranker = LGBMRanker(
        objective="lambdarank",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        label_gain=lgb_gain,
    )
    t0 = time.time()
    lgb_ranker.fit(X_train_ord, y_train_gain, group=group_train)
    fit_time = time.time() - t0

    plot_and_save_feature_importance(
        lgb_ranker,
        feature_cols,
        "LightGBM",
        task_key,
        VISUALIZATION_DIR,
        suffix="_holdout",
        feature_label_map=FEATURE_NAME_JP,
        xlabel_jp="重要度（％）",
        ylabel_jp="特徴量",
    )

    y_pred_scores = lgb_ranker.predict(X_test_ord)
    df_eval = pd.DataFrame({"_qid_ordered": qid_test_ord.values, "true_gain": y_test_gain, "pred_score": y_pred_scores})
    metrics = eval_metrics_per_query(df_eval, score_col="pred_score", rel_col_as_gain="true_gain")
    print(
        "[LightGBM]  "
        f"NDCG@1={metrics['NDCG@1']:.4f}  NDCG@3={metrics['NDCG@3']:.4f}  "
        f"NDCG@5={metrics['NDCG@5']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}  "
        f"NDCG@ALL={metrics['NDCG@ALL']:.4f}  "
        f"Hit@1={metrics['Hit@1']:.4f}  Hit@3={metrics['Hit@3']:.4f}  Hit@5={metrics['Hit@5']:.4f}  "
        f"Hit@10={metrics['Hit@10']:.4f}  Hit@ALL={metrics['Hit@ALL']:.4f}  "
        f"ケンドール={metrics['ケンドール']:.4f}  スピアマン={metrics['スピアマン']:.4f}  ピアソン={metrics['ピアソン']:.4f}  "
        f"学習時間={fit_time:.2f}秒",
    )
    results_task["LightGBM"] = {
        "NDCG@1": metrics["NDCG@1"],
        "NDCG@3": metrics["NDCG@3"],
        "NDCG@5": metrics["NDCG@5"],
        "NDCG@10": metrics["NDCG@10"],
        "NDCG@ALL": metrics["NDCG@ALL"],
        "Hit@1": metrics["Hit@1"],
        "Hit@3": metrics["Hit@3"],
        "Hit@5": metrics["Hit@5"],
        "Hit@10": metrics["Hit@10"],
        "Hit@ALL": metrics["Hit@ALL"],
        "ケンドール": metrics["ケンドール"],
        "スピアマン": metrics["スピアマン"],
        "ピアソン": metrics["ピアソン"],
        "fit_time_sec": fit_time,
        "num_queries": metrics["num_queries"],
    }
    preds_task["LightGBM"] = y_pred_scores

    # XGBoost
    xgb_ranker = XGBRanker(
        objective="rank:ndcg",
        ndcg_exp_gain=False,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    t0 = time.time()
    xgb_ranker.fit(X_train_ord, y_train_gain, group=group_train)
    fit_time = time.time() - t0

    plot_and_save_feature_importance(
        xgb_ranker,
        feature_cols,
        "XGBoost",
        task_key,
        VISUALIZATION_DIR,
        suffix="_holdout",
        feature_label_map=FEATURE_NAME_JP,
        xlabel_jp="重要度（％）",
        ylabel_jp="特徴量",
    )

    y_pred_scores = xgb_ranker.predict(X_test_ord)
    df_eval = pd.DataFrame({"_qid_ordered": qid_test_ord.values, "true_gain": y_test_gain, "pred_score": y_pred_scores})
    metrics = eval_metrics_per_query(df_eval, score_col="pred_score", rel_col_as_gain="true_gain")
    print(
        "[XGBoost]   "
        f"NDCG@1={metrics['NDCG@1']:.4f}  NDCG@3={metrics['NDCG@3']:.4f}  "
        f"NDCG@5={metrics['NDCG@5']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}  "
        f"NDCG@ALL={metrics['NDCG@ALL']:.4f}  "
        f"Hit@1={metrics['Hit@1']:.4f}  Hit@3={metrics['Hit@3']:.4f}  Hit@5={metrics['Hit@5']:.4f}  "
        f"Hit@10={metrics['Hit@10']:.4f}  Hit@ALL={metrics['Hit@ALL']:.4f}  "
        f"ケンドール={metrics['ケンドール']:.4f}  スピアマン={metrics['スピアマン']:.4f}  ピアソン={metrics['ピアソン']:.4f}  "
        f"学習時間={fit_time:.2f}秒",
    )
    results_task["XGBoost"] = {
        "NDCG@1": metrics["NDCG@1"],
        "NDCG@3": metrics["NDCG@3"],
        "NDCG@5": metrics["NDCG@5"],
        "NDCG@10": metrics["NDCG@10"],
        "NDCG@ALL": metrics["NDCG@ALL"],
        "Hit@1": metrics["Hit@1"],
        "Hit@3": metrics["Hit@3"],
        "Hit@5": metrics["Hit@5"],
        "Hit@10": metrics["Hit@10"],
        "Hit@ALL": metrics["Hit@ALL"],
        "ケンドール": metrics["ケンドール"],
        "スピアマン": metrics["スピアマン"],
        "ピアソン": metrics["ピアソン"],
        "fit_time_sec": fit_time,
        "num_queries": metrics["num_queries"],
    }
    preds_task["XGBoost"] = y_pred_scores

    # CatBoost
    train_pool = Pool(data=X_train_ord, label=y_train_gain, group_id=qid_train_ord.astype(str).values)
    test_pool = Pool(data=X_test_ord, label=y_test_gain, group_id=qid_test_ord.astype(str).values)

    cat_ranker = CatBoostRanker(
        iterations=600,
        learning_rate=0.05,
        depth=8,
        loss_function="YetiRank",
        eval_metric="NDCG:top=10",
        random_seed=42,
        verbose=False,
    )
    t0 = time.time()
    cat_ranker.fit(train_pool, eval_set=test_pool)
    fit_time = time.time() - t0

    plot_and_save_feature_importance(
        cat_ranker,
        feature_cols,
        "CatBoost",
        task_key,
        VISUALIZATION_DIR,
        suffix="_holdout",
        feature_label_map=FEATURE_NAME_JP,
        xlabel_jp="重要度（％）",
        ylabel_jp="特徴量",
        catboost_data=train_pool,
    )

    y_pred_scores = cat_ranker.predict(test_pool).ravel()
    df_eval = pd.DataFrame({"_qid_ordered": qid_test_ord.values, "true_gain": y_test_gain, "pred_score": y_pred_scores})
    metrics = eval_metrics_per_query(df_eval, score_col="pred_score", rel_col_as_gain="true_gain")
    print(
        "[CatBoost]  "
        f"NDCG@1={metrics['NDCG@1']:.4f}  NDCG@3={metrics['NDCG@3']:.4f}  "
        f"NDCG@5={metrics['NDCG@5']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}  "
        f"NDCG@ALL={metrics['NDCG@ALL']:.4f}  "
        f"Hit@1={metrics['Hit@1']:.4f}  Hit@3={metrics['Hit@3']:.4f}  Hit@5={metrics['Hit@5']:.4f}  "
        f"Hit@10={metrics['Hit@10']:.4f}  Hit@ALL={metrics['Hit@ALL']:.4f}  "
        f"ケンドール={metrics['ケンドール']:.4f}  スピアマン={metrics['スピアマン']:.4f}  ピアソン={metrics['ピアソン']:.4f}  "
        f"学習時間={fit_time:.2f}秒",
    )
    results_task["CatBoost"] = {
        "NDCG@1": metrics["NDCG@1"],
        "NDCG@3": metrics["NDCG@3"],
        "NDCG@5": metrics["NDCG@5"],
        "NDCG@10": metrics["NDCG@10"],
        "NDCG@ALL": metrics["NDCG@ALL"],
        "Hit@1": metrics["Hit@1"],
        "Hit@3": metrics["Hit@3"],
        "Hit@5": metrics["Hit@5"],
        "Hit@10": metrics["Hit@10"],
        "Hit@ALL": metrics["Hit@ALL"],
        "ケンドール": metrics["ケンドール"],
        "スピアマン": metrics["スピアマン"],
        "ピアソン": metrics["ピアソン"],
        "fit_time_sec": fit_time,
        "num_queries": metrics["num_queries"],
    }
    preds_task["CatBoost"] = y_pred_scores

    metrics_all[task_key] = results_task
    pred_store[task_key] = preds_task


# 全データで学習
if flag:
    print("\n【全データで学習して保存】\n")

    preprocessor_full = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
            ("num", StandardScaler(), numeric_wo_embed),
        ],
        remainder="drop",
    )
    preprocessor_full.fit(X_all)
    X_all_t = preprocessor_full.transform(X_all)

    rel_all_dict = {k: v for k, v in rel_cols.items()}

    X_all_ord, qid_all_ord, rel_all_ordered, _ = order_by_query(
        X_all_t,
        group_series.reset_index(drop=True),
        rel_all_dict,
        original_index=X_all.index,
    )

    group_all = [len(g) for _, g in pd.Series(qid_all_ord).groupby(qid_all_ord, sort=False)]

    saved_paths = {}
    for task_key in ["precision", "recall", "f1", "time"]:
        y_all_gain = rel_to_gain(rel_all_ordered[task_key])
        max_gain = int(max(y_all_gain.max(), 64))
        lgb_gain = list(range(max_gain + 1))

        model_paths = {}

        # LightGBM
        lgb_ranker = LGBMRanker(
            objective="lambdarank",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            label_gain=lgb_gain,
        )
        lgb_ranker.fit(X_all_ord, y_all_gain, group=group_all)

        plot_and_save_feature_importance(
            lgb_ranker,
            feature_cols,
            "LightGBM",
            task_key,
            VISUALIZATION_DIR,
            suffix="_final",
            feature_label_map=FEATURE_NAME_JP,
            xlabel_jp="重要度（％）",
            ylabel_jp="特徴量",
        )

        bundle = {
            "preprocessor": preprocessor_full,
            "ranker": lgb_ranker,
            "feature_cols_in_order": feature_cols,
            "task": task_key,
            "rank_label": rank_cols_map[task_key],
            "label_convention": "rel=1..64 (best=1,worst=64); trained_on_gain=65-rel",
        }
        out_path = ARTIFACT_DIR / "models" / f"rank_{task_key}_LightGBM.joblib"
        joblib.dump(bundle, out_path)
        model_paths["LightGBM"] = str(out_path)
        print(f"{task_key} - LightGBM: 保存 -> {out_path.name}")

        # XGBoost
        xgb_ranker = XGBRanker(
            objective="rank:ndcg",
            ndcg_exp_gain=False,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        xgb_ranker.fit(X_all_ord, y_all_gain, group=group_all)

        plot_and_save_feature_importance(
            xgb_ranker,
            feature_cols,
            "XGBoost",
            task_key,
            VISUALIZATION_DIR,
            suffix="_final",
            feature_label_map=FEATURE_NAME_JP,
            xlabel_jp="重要度（％）",
            ylabel_jp="特徴量",
        )

        bundle = {
            "preprocessor": preprocessor_full,
            "ranker": xgb_ranker,
            "feature_cols_in_order": feature_cols,
            "task": task_key,
            "rank_label": rank_cols_map[task_key],
            "label_convention": "rel=1..64 (best=1,worst=64); trained_on_gain=65-rel; ndcg_exp_gain=False for XGB",
        }
        out_path = ARTIFACT_DIR / "models" / f"rank_{task_key}_XGBoost.joblib"
        joblib.dump(bundle, out_path)
        model_paths["XGBoost"] = str(out_path)
        print(f"{task_key} - XGBoost: 保存 -> {out_path.name}")

        # CatBoost
        cat_pool_all = Pool(data=X_all_ord, label=y_all_gain, group_id=pd.Series(qid_all_ord).astype(str).values)
        cat_ranker = CatBoostRanker(
            iterations=600,
            learning_rate=0.05,
            depth=8,
            loss_function="YetiRank",
            random_seed=42,
            verbose=False,
        )
        cat_ranker.fit(cat_pool_all)

        plot_and_save_feature_importance(
            cat_ranker,
            feature_cols,
            "CatBoost",
            task_key,
            VISUALIZATION_DIR,
            suffix="_final",
            feature_label_map=FEATURE_NAME_JP,
            xlabel_jp="重要度（％）",
            ylabel_jp="特徴量",
            catboost_data=cat_pool_all,
        )

        bundle = {
            "preprocessor": preprocessor_full,
            "ranker": cat_ranker,
            "feature_cols_in_order": feature_cols,
            "task": task_key,
            "rank_label": rank_cols_map[task_key],
            "label_convention": "rel=1..64 (best=1,worst=64); trained_on_gain=65-rel; CatBoost=YetiRank",
        }
        out_path = ARTIFACT_DIR / "models" / f"rank_{task_key}_CatBoost.joblib"
        joblib.dump(bundle, out_path)
        model_paths["CatBoost"] = str(out_path)
        print(f"{task_key} - CatBoost: 保存 -> {out_path.name}")

        saved_paths[task_key] = model_paths

    # メタデータ
    registry = {
        "type": "learning-to-rank",
        "tasks": ["precision", "recall", "f1", "time"],
        "models": ["LightGBM", "XGBoost", "CatBoost"],
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "feature_cols_in_order": feature_cols,
        "holdout_metrics": metrics_all,
        "saved_models": saved_paths,
        "split": {"type": "GroupShuffleSplit", "group_key": "question", "test_size": 0.2, "random_state": 42},
        "label_convention": (
            "rel=1..64 (best=1,worst=64); trained_on_gain=65-rel; ndcg_exp_gain=False for XGB; CatBoost=YetiRank"
        ),
        "metrics": [
            "NDCG@1",
            "NDCG@3",
            "NDCG@5",
            "NDCG@10",
            "NDCG@ALL",
            "Hit@1",
            "Hit@3",
            "Hit@5",
            "Hit@10",
            "Hit@ALL",
            "ケンドール",
            "スピアマン",
            "ピアソン",
        ],
    }

    (ARTIFACT_DIR / "registry").mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_DIR / "registry" / "rank_model_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    print("\n保存メタデータ: artifacts/registry/rank_model_registry.json")

print("\n完了")
