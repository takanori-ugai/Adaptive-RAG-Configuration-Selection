# 論文改訂計画書 (Paper Update Plan)

本ドキュメントは、[Review.txt](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Review.txt) に記載された査読コメント（査読者 #1、査読者 #2）を分析し、人工知能学会論文誌の技術論文として採択されるレベルに引き上げるための具体的な改訂計画をまとめたものです。

---

## 1. 査読結果の要約と改訂の方向性

不採択（判定：不採択・再投稿推奨）の主たる理由は以下の2点に集約されます。
1. **新規性の立証不足**: 近接する先行研究（CARROT, IUM/uRAG, RAGRouter, AdaGCRAG）との差異が整理されておらず、「文書特徴の考慮」や「精度・コストの統合学習」に関する導入部の主張が過大であると指摘されました。
2. **有用性の立証不足**: 自明な固定ベースライン（最頻出設定）や既存の適応的RAG手法と比較した際の有意な優位性（Pareto改善）が示されておらず、評価指標の妥当性や統計的有意性の検証も欠けています。

**改訂の方向性**:
要素技術の組み合わせに留まらない本研究独自の貢献を明確に再定義し、既存の代表的な適応的RAG手法（Adaptive-RAG等）との比較実験、標準的QA指標（Exact Match / F1）による再評価、統計的有意差検定、および累積コストの適正な計上を追加します。

---

## 2. 指摘事項ごとの対応計画

### 2.1 新規性・関連研究に関する改善項目 (Major)

*   **指摘内容 (査読者 #1)**: 
    *   近接研究（CARROT, IUM/uRAG, RAGRouter, AdaGCRAG）が存在するため、「文書特徴の考慮が不足」「精度とコストの統合学習が未確立」という主張は過大。
    *   先行研究との差異を明確に整理し、自手法の独自性を具体的に限定して記述すること。
*   **対応計画**:
    1.  [sec1.tex](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec1.tex) における過大な主張（「未確立」等の表現）を緩和し、以下のように独自性を限定した記述に改めます。
        *   推論時にLLMを要しない完全オフラインのlearning-to-rankアプローチである点。
        *   Graph RAGとNon-Graph RAGを横断し、かつチャンクサイズ等のハイパーパラメータをも含む96通りの異種構成空間を統一的に順位付けする設計。
        *   適合率・再現率・F1・実行時間の多目的をユーザ重みで切り替え可能な統合設計。
    2.  [sec2.tex](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec2.tex)（関連研究）に CARROT, IUM/uRAG, RAGRouter, AdaGCRAG を引用追加し、手法の差分を示す比較表を新規追加します。

### 2.2 有用性・実験評価に関する改善項目 (Major)

*   **指摘内容 (査読者 #1 & #2)**:
    *   (1) 先行の適応的RAG手法（例: 質問複雑度で分岐する Adaptive-RAG）との比較実験がない。
    *   (2) 表3において「最頻出設定」と提案手法（提案@1）の実測性能がほぼ同等であり、固定構成に対する優位性（Pareto改善）が示されていない。
    *   (3) 評価項目(iii)（上位K個から選択する運用）において、K個を試行した「累積実行時間（総コスト）」が計上されておらず、追加コストに見合う精度向上かが判断できない。
    *   (4) アブレーションにおいて、文書特徴を追加してもNDCG@1が66.84から68.52へ1.7ポイントしか改善せず、一部の組合せではC単独を下回るなど、文書特徴の寄与が限定的。
    *   (5) 評価データが80事例と小規模であり、有意差検定がない。
    *   (6) 評価指標がBERTScoreのみであり、意味的類似性は測れるが事実性や標準QA指標（Exact Match / token-level F1）との一致度が不明。
*   **対応計画**:
    1.  **既存手法との比較追加**: 質問の難易度に応じてルーティングを行う代表的な手法である **Adaptive-RAG (Jeong et al., 2024)** をベースラインとして実装・実験し、性能比較データを追加します。
    2.  **Pareto改善の再評価**: 単一の最頻出設定に対して、提案手法が「同等コストで高いF1値」または「同等F1値で低コスト」を達成できているかをParetoフロンティアの観点からプロット等を用いて可視化・検証します。
    3.  **累積実行時間の計上**: 評価項目(iii)（提案@3, 提案@5）における実行時間について、採用された構成の単一時間だけでなく、**K個全てのRAG構成を試行するのに要した累積時間（総実行時間）** を算出して表に追記します。
    4.  **文書特徴の寄与に関する考察**: アブレーション結果を率直に議論し、どのような質問・文書タイプにおいて文書特徴が機能するのか（例: 文書長が極端に長い場合や、固有表現が多く含まれるマルチホップ質問など）を定性的・定量的に分析し、限定性について正直に議論します。
    5.  **統計的有意差検定の実施**: 評価データ（80事例）における提案手法とベースライン（最頻出設定、Adaptive-RAGなど）の差について、ブートストラップ法または対応のある検定（Wilcoxon符号付き順位検定など）を適用し、p値を算出・報告します。
    6.  **標準QA評価指標の併記**: BERTScoreに加え、QAタスクの標準指標である **Exact Match (EM)** および **token-level F1** を計測し、実験結果に追加します。

### 2.3 定式化・妥当性に関する改善項目 (Medium)

*   **指摘内容 (査読者 #1 & #2)**:
    *   (7) 式(1)の集約方式（逆数ランクの単純加算）の妥当性を示すアブレーション（線形和や対数和等との比較）がない。
    *   (8) 表4の提案@3において、F1値重視（再現率59.58）が再現率重視（59.37）より高いという挙動の不整合がある。また、上位K候補がどの重み設定で抽出されたか不明。
    *   (9) 質問タイプ分類が英語ルールベース（is/can, what/who, or等）に依存している。日本語や専門ドメインでの一般化可能性についての議論がない。
    *   (10) ランダム選択の数値（適合率0.06%, 再現率23.37%）が不自然に乖離している。
*   **対応計画**:
    1.  **集約関数のアブレーション**: 式(1)の単純加算、対数順位加算（NDCG的な割引）、および予測値（生スコア）の線形結合などを比較するアブレーション実験を行い、結果を [sec5.4](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec5.tex#L133) に追記します。
    2.  **重み設定の妥当性評価**: 査読者#2の「重みを変化させた結果が意図通り変化するか」という疑問に対し、重み（例: $W_{\mathrm{time}}$）を $0.0$ から $1.0$ まで連続的（あるいは段階的）に変化させた際のF1値と実行時間の変化をプロットし、線形結合の妥当性とトレードオフ制御の有効性を実証します。また、上位K候補の生成プロセス（第一の設定である $F_1$-時間バランス重視の重みを用いて上位K件を選定した後に、各重視方針を適用した等）を [sec4.1](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec4.tex#L20) に明記します。
    3.  **不整合とランダム挙動の調査**: 
        *   提案@3におけるF1値重視と再現率重視の数値の逆転について、原因（データセット内の分散や、予測順位と実測値の相関の限界）を分析し、本文に注釈を加えます。
        *   ランダム選択の挙動（多くの構成がエラーやタイムアウト等で失敗した可能性、あるいは計算方法の定義）を再確認し、本文で解説します。
    4.  **言語依存性の明記**: ルールベース分類が英語依存であることを [sec5.8](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec5.8)（本研究の限界）に明記し、日本語環境（例: MeCab等を用いた形態素解析や疑問詞判定）や専門ドメインへの適用方法を今後の課題として議論します。

### 2.4 提示・記述の軽微な修正 (Minor)

*   **指摘内容 (査読者 #1)**:
    *   (11) 学会口頭発表版からの技術論文としての追加差分（増分）を明示すること。
    *   (12) 英文抄録のスペルミス・校正不足（`base don` 等）。数値の不一致（本文5.3節の `59.46` と 表4の `59.58`）。
*   **対応計画**:
    1.  [sec1.tex](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec1.tex) の末尾に、学会版 [anaguchi26] に対する具体的な追加要素（詳細なアブレーション分析、特徴量の重要度評価、エラー分析を含む定性評価の追加など）を明記します。
    2.  [main.tex](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/main.tex) の英文抄録 (`summary`環境) を校正し、改行由来の語結合等の誤記を修正します。
    3.  [sec5.3](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec5.tex#L96) における本文中の再現率の数値（59.46）を、表4の正確な数値（59.58）に修正・統一します。

---

## 3. LaTeXソースファイル修正マッピング

| 修正対象ファイル | 修正予定セクション | 修正内容の詳細 |
| :--- | :--- | :--- |
| [main.tex](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/main.tex) | `summary` (英文抄録) | 改行・単語結合のタイポ修正 (`base don` $\rightarrow$ `based on`等) |
| [sec1.tex](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec1.tex) | 1. はじめに | 新規性に関する過大主張の緩和、提案手法の強み（完全オフライン、96通りの異種空間等）の再整理、学会発表版からの差分の追加 |
| [sec2.tex](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec2.tex) | 2. 関連研究 | CARROT, IUM/uRAG, RAGRouter, AdaGCRAG の追加、特徴比較表の作成 |
| [sec3.tex](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec3.tex) | 3. 提案手法 | 式(1)の集約方式の導入背景、上位K件の抽出プロセスの形式化 |
| [sec4.tex](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec4.tex) | 4. 評価実験の方法 | Adaptive-RAGベースラインの追加、標準指標（EM, token-F1）の導入、有意差検定（Wilcoxon）の説明、評価項目(iii)での累積コスト定義の修正 |
| [sec5.tex](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec5.tex) | 5. 実験結果および考察 | 既存手法（Adaptive-RAG）との比較表（表3, 4）の更新、有意差検定結果の追加、累積実行時間の追加、集約方式アブレーション結果の追加、重みスイープ分析の追加、数値不整合の修正 (59.46 $\rightarrow$ 59.58) |
| [sec6.tex](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/sec6.tex) | 6. おわりに | 改訂した結果に基づき、最頻出設定や既存手法に対する優位性（Pareto改善）を考慮した適正な結言への修正 |
| [references.bib](file:///home/ugai/Adaptive-RAG-Configuration-Selection/Paper/Tex/references.bib) | 文献データベース | 新規に追加する先行研究文献（CARROT, IUM, uRAG, RAGRouter, AdaGCRAG, Adaptive-RAG）のbibtexの追加 |
