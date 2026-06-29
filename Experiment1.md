# Adaptive-RAGとの比較評価に関する追加実験計画書 (Additional Experiment Plan with Adaptive-RAG)

本ドキュメントは、[Update-Plan.md](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Update-Plan.md) の方針に基づき、先行研究の代表的な適応的RAG手法である **Adaptive-RAG (Jeong et al., NAACL 2024)** を本リポジトリの評価フレームワークにベースラインとして組み込み、比較評価を行うための追加実験計画をまとめたものです。

実装には、公式実装である [starsuzi/Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG) のロジックおよび学習済み分類器を利用します。

---

## 1. 実験の目的
本研究で提案する「完全オフラインかつ多目的重み付きのランキングモデルによる適応的RAG構成選択」と、クエリの複雑度分類に基づいてルーティングを行う既存手法「Adaptive-RAG」を比較します。
これにより、**「単純なRAG手法間のルーティングのみを行う既存手法」に対する「ハイパーパラメータをも含む広範な構成空間の多目的順位付けを行う提案手法」の優位性（Pareto改善）** を定量的および定性的に評価・立証することを目的とします。

---

## 2. 実験設計とAdaptive-RAGの統合

### 2.1. クエリ複雑度分類器の導入
Adaptive-RAGの基幹部分である、質問の複雑度を予測する分類器（Classifier）を導入します。
*   **分類器のモデル**: `starsuzi/Adaptive-RAG` で用いられている、質問文から `{0: No-Retrieval, 1: Single-Step, 2: Multi-Step}` の3クラスを予測する分類器（例: T5-large または DeBERTa-v3-small）を使用します。
*   **学習データ**: 本リポジトリの訓練用質問データ（400事例）を使用します。
    *   SQuADデータセット由来の質問を `1: Single-Step`、HotpotQA/2WikiMultihopQA/MuSiQue由来の質問を `2: Multi-Step` と擬似ラベル（Gold Labels）を付与して分類器を微調整（Fine-tuning）します。

### 2.2. 複雑度からRAG構成へのマッピング
分類器の出力クラスに応じて、本リポジトリの96通りのRAG構成空間から、それぞれを代表する構成を割り当ててシミュレートします。
本リポジトリのテストデータ [test_all_ranks.csv](file:///home/ugai/Adaptive-RAG-Configuration-Selection/data/statistics/test_all_ranks_stats.csv) には全構成の実測値が既に含まれているため、以下のマッピングによって、新たなRAG実行コストをかけることなく評価が可能です。

1.  **`0: No-Retrieval` (検索なし)**:
    *   本構成空間で最も実行時間が短く、追加検索を行わない構成（擬似的に NaiveRAG の最小チャンクサイズ設定、またはコンテキストを空にして直接LLMに回答させる構成）をマッピングします。
2.  **`1: Single-Step` (単一文書検索)**:
    *   本リポジトリの固定ベースラインである「標準設定」または「最頻出設定」をマッピングします。
        *   代表例: `NaiveRAG, stuff, 500, 50, llama3.1:8b, nomic-embed-text`
3.  **`2: Multi-Step` (複数文書検索・統合)**:
    *   グラフベースRAGのローカル検索またはグローバル検索構成をマッピングします。
        *   代表例: `GraphRAG, Local, 500, 100, llama3.1:8b, nomic-embed-text`

---

## 3. 実装手順

### 3.1. 依存ライブラリの追加
[requirements.txt](file:///home/ugai/Adaptive-RAG-Configuration-Selection/requirements.txt) に分類器の構築に必要な `transformers` や `accelerate` などのライブラリを追記します。

### 3.2. スクリプトの新規作成
`inference/` ディレクトリ配下に `adaptive_rag_baseline.py` を新規作成します。

```python
# inference/adaptive_rag_baseline.py の擬似コード
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict_complexity(questions, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # 3クラス分類 (0: No-Retrieval, 1: Single-Step, 2: Multi-Step) の推論
    ...
    return predictions

def main():
    # 1. テストデータの読み込み
    test_df = pd.read_csv("../data/statistics/test_all_ranks_stats.csv")
    
    # 2. 各質問に対する複雑度予測
    unique_qs = test_df["question"].unique()
    complexities = predict_complexity(unique_qs, "starsuzi/Adaptive-RAG-classifier")
    q_to_complexity = dict(zip(unique_qs, complexities))
    
    # 3. 複雑度に基づくRAG構成の割り当て
    selected_rows = []
    for q, g in test_df.groupby("question"):
        comp = q_to_complexity[q]
        if comp == 0:
            # 最も低コストな設定を選択
            config = g.loc[g["time"].idxmin()]
        elif comp == 1:
            # NaiveRAG (Single-Step)
            config = g[(g["rag_name"] == "NaiveRAG") & (g["chain_type"] == "stuff")].iloc[0]
        else:
            # GraphRAG (Multi-Step)
            config = g[(g["rag_name"] == "GraphRAG")].iloc[0]
        selected_rows.append(config)
        
    # 4. 指標の集計と保存
    result_df = pd.DataFrame(selected_rows)
    result_df.to_csv("../artifacts/results/adaptive_rag_baseline_results.csv", index=False)
```

### 3.3. 評価指標の計測
[selection_rag.py](file:///home/ugai/Adaptive-RAG-Configuration-Selection/inference/selection_rag.py) の `run_score_summary` と同様の集計ロジックを用いて、Adaptive-RAG適用時の「適合率、再現率、F1値 (BERTScore / EM / Token-F1)、実行時間」の平均値を算出します。

---

## 4. 比較評価の項目と分析方法

### 4.1. 比較対象
以下の手法の間で性能を比較します。
1.  **標準設定 (LangChain Default)**: すべての質問に NaiveRAG (CS=250, CO=50) を適用
2.  **最頻出設定 (Most Frequent Configuration)**: すべての質問に NaiveRAG (CS=500, CO=50) を適用
3.  **Adaptive-RAG (Jeong et al., 2024)**: クエリ複雑度分類に基づく3つのRAG戦略ルーティング
4.  **提案手法 (Proposal)**: 
    *   $F_1$-時間バランス重視 ($W_{\text{f_1}}=0.5, W_{\text{time}}=0.5$)
    *   精度重視 ($W_{\text{f_1}}=1.0$)
    *   時間重視 ($W_{\text{time}}=1.0$)
5.  **オラクル (Oracle)**: 事後的な理論上限値

### 4.2. 評価指標の追加
[Update-Plan.md](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Update-Plan.md) の方針通り、以下の指標を用いて比較を行います。
*   **回答精度**: BERTScore ($F_1$) / Exact Match (EM) / Token-level F1
*   **実行コスト**: 平均実行時間 (秒)
*   **統計的有意性**: 提案手法と Adaptive-RAG の回答精度（EMおよびF1値）の差について、Wilcoxon符号付き順位検定による有意差検定を行います。

### 4.3. 期待される結果と考察のポイント
*   **精度とコストのParetoフロンティア**: Adaptive-RAGは3つの離散的なルーティング先しか選べないため、精度やコストの柔軟な制御が不可能です。提案手法は重み $W$ を変化させることで、Adaptive-RAGを上回る効率的なParetoフロンティア（同等の時間でより高い精度、または同等の精度でより短い実行時間）を描くことが期待されます。
*   **ハイパーパラメータ最適化の効果**: 単にRAG手法を切り替えるだけでなく、質問/文書の特徴量（文書トークン数や類似度など）から、チャンクサイズやモデルの組み合わせといった「構成内のミクロなハイパーパラメータ」を最適化することが、回答精度の差（EM / F1の向上）にどのように貢献しているかをアブレーションを通じて実証します。
