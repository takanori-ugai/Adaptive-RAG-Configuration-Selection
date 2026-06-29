# 先行・関連研究と提案手法の比較表 (Comparison of RAG Optimization Methods)

本ドキュメントは、[Update-Plan.md](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Update-Plan.md) および [List.md](file:///home/ugai/Adaptive-RAG-Configuration-Selection/List.md) で議論されている先行・関連研究（CARROT, IUM/uRAG, RAGRouter, AdaGCRAG, Adaptive-RAG）と、本論文の提案手法との特徴比較をまとめたものです。

---

## 1. 比較一覧表

| 手法名 | 主な目的 (Primary Objective) | 最適化対象 (Action/Search Space) | 選択アルゴリズム (Decision Algorithm) | コスト・精度の多目的制御 (Multi-objective) | 主な特徴と制限 (Key Characteristics & Limitations) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **提案手法**<br>(Our Method) | コストと精度のトレードオフを考慮したRAG構成の適応的選択 | RAG手法、検索設定、CS、CO、LLM(GM)、EMの組み合わせ（[全96通り](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec3.tex#L66)） | クエリと文書特徴量（7種）を入力とする完全オフライン学習のランキングモデル（LightGBM等） | **あり**<br>適合率・再現率・F1・時間の予測順位の逆数をユーザー定義の[重み $W$ で統合](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec3.tex#L162) | 異種RAG（Graph/Non-Graph）とミクロなハイパーパラメータを統一順位付け。推論時にLLM不要で超高速。 |
| **CARROT**<br>(ICDE 2026) | 検索チャンクの組合せ・順序のコスト制約下での最適化 | 検索された複数チャンクの並び順および選択、構成エージェント設定 | モンテカルロ木探索（MCTS）を用いたオンライン状態探索 | **あり**<br>動的に設定されたコスト制約下での回答品質の最大化 | チャンク間の文脈依存性や「情報追加による品質低下」をオンラインで解決。探索による推論時オーバーヘッドが存在。 |
| **IUM / uRAG**<br>(ICTIR '25 / SIGIR '24) | 複数RAGモデルの検索結果リストに対するLearning-to-Rank (LTR) | 複数の外部検索モジュールから取得されたドキュメントのランキングと統合 | 反復的効用最大化（Iterative Utility Maximization）に基づくランク学習 | **あり**<br>クライアント側のコスト・精度要件に応じた効用最大化 | 文書の順位付け（Retrieval段階）のマージと最適化に特化。LLMやチャンク分割等の下流パラメータ選択は対象外。 |
| **RAGRouter**<br>(NeurIPS 2025) | 取得文書とクエリに適した生成LLMの動的ルーティング | 複数の候補生成LLM（能力やコストの異なるモデル）の選択 | 対照学習で事前学習された複合知識ベクトル表現に基づくルーティングモデル | **あり**<br>モデルの処理コストと応答性能のトレードオフ最適化 | 文書取得後に「その文書をLLMが処理できるか（有効知識量）」を明示的にモデリングしてLLMを決定する。 |
| **AdaGCRAG**<br>(2025) | 軽量RAGのための適応的グラフチャンク検索の最適化 | グラフベースRAGにおける検索対象グラフ（エンティティ・リレーション）の範囲・深さ | クエリ特徴に基づく適応的グラフ検索戦略 | **間接的**<br>グラフ抽出コストの抑制と精度維持のバランス | 知識グラフのエンティティ関係の適応的抽出に特化。小規模LLM環境での高効率化が主眼。 |
| **Adaptive-RAG**<br>(NAACL 2024) | 質問の複雑度に応じたRAG検索戦略の動的ルーティング | 3つのマクロな検索戦略<br>（No-Retrieval / Single-Step / Multi-Step） | 軽量な分類器（T5やDeBERTa等）による質問複雑度（3クラス）の予測 | **なし**<br>質問の複雑度に紐づく固定ルールによる戦略分岐 | 質問自体の難易度に応じてRAG処理フロー自体を切り替える。実装がシンプルで判定オーバーヘッドが極小（[詳細計画](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Experiment1.md)参照）。 |

---

## 2. 提案手法のポジショニングと差別化ポイント

*   **異種ハイパーパラメータを含む一括最適化**:
    *   既存手法の多くが「RAG手法の切り替え」のみ（Adaptive-RAG）、または「LLMの切り替え」のみ（RAGRouter）、または「ドキュメント順位」のみ（uRAG）に特化しているのに対し、提案手法は**RAGの種類からチャンクサイズ、使用モデル（GM, EM）に及ぶ多層的なハイパーパラメータ（96通り）を一括で順位付け**できます。
*   **動的な多目的優先度制御（プラグマティックな適用）**:
    *   Adaptive-RAG等のようにハードコードされたルールで分岐するのではなく、同一のランキング予測結果を用いながら、**実行時の制約（例: レイテンシ優先、または正確性優先）に合わせてユーザーが重み $W$ を切り替えるだけで動的に振る舞いを変えられる**点に優位性があります。
