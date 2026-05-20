# CEINN 研究執行計畫

**研究題目**：CEINN: Causal Economics-Informed Neural Networks for Sequential Recommendation  
**文件類型**：完整研究執行計畫（Full Execution Plan）  

---

## 計畫總覽

本計畫將 CEINN 的完整研究流程解構為八個大執行階段，從原始資料的匯入與驗證，一路延伸至最終的實驗結果討論與論文撰寫準備。每個大階段均列出**關鍵任務（Key Tasks）**、**輸出物件（Deliverables）**、**品質閘（Quality Gate）**，以及與專案目錄的對應關係。

```
Phase 1  ──▶  Phase 2  ──▶  Phase 3  ──▶  Phase 4
資料匯入        前處理與         模型架構         損失函數
與完整性        特徵工程         實作             與訓練流程
驗證                                             實作

Phase 5  ──▶  Phase 6  ──▶  Phase 7  ──▶  Phase 8
基線模型        CEINN 主         消融實驗         結果分析
建立與          模型訓練         與超參數         與討論
評估            與驗證           搜索
```

---

## Phase 1：原始資料匯入、解析與完整性驗證

> **目標**：確保兩個資料集的原始檔案能被正確讀取與解析，並對照 EDA 報告所記載的統計數字完成完整性驗證，在任何前處理步驟開始之前，確立乾淨的資料基線。

### 1.1 Amazon Beauty 原始資料解析

**任務 1.1.1：讀取三份 JSON 原始檔案**

`data/raw/` 下包含三份不同結構的檔案：

- `reviews_Beauty.json`：評論主體，每行一筆 JSON，包含 `reviewerID`、`asin`、`overall`（1–5 整數評分）、`unixReviewTime`
- `Beauty_5.json`：已過 5-core 過濾的評論子集（EDA 報告的主要分析來源）
- `meta_Beauty.json`：商品元資料，包含 `asin`、`price`、`salesRank`（dict，含品類鍵值）、`categories`（巢狀列表）、`brand`

確認三份檔案的讀取方式（逐行 JSON 解析 vs. 標準 JSON array），以及是否存在格式不一致的邊緣案例（如 `salesRank` 欄位可能以字串或 `null` 出現）。

**任務 1.1.2：完整性驗證**

對照 EDA 報告的統計數字進行逐項核驗：

| 驗證項目 | EDA 預期值 | 容許誤差 |
|---------|-----------|---------|
| 互動總筆數 | 198,502 | 0 |
| 獨立使用者數 | 22,363 | 0 |
| 獨立商品數（reviews 中） | 12,101 | 0 |
| 5 星評分比例 | 57.70% | ±0.1% |
| Price 覆蓋率 | 95.17% | ±0.5% |
| SalesRank 任意品類覆蓋率 | 98.2% | ±0.5% |
| Beauty 品類 SalesRank 覆蓋率 | 88.3% | ±0.5% |
| 品牌覆蓋率 | 82.8% | ±0.5% |
| 葉節點類別數 | 222 | 0 |

若任何數值超出容許誤差，需回頭確認讀取邏輯（如重複評論的去重策略、巢狀 categories 的葉節點萃取方式）。

**任務 1.1.3：時間戳記驗證**

確認 `unixReviewTime` 欄位的時間跨度落在 EDA 所記載的 2002-06-12 至 2014-07-23 範圍（4,424 天），並確認時間戳記為 Unix epoch（秒）而非毫秒，排除因單位錯誤造成的時序排序失敗。

### 1.2 MovieLens 10M 原始資料解析

**任務 1.2.1：讀取 `.dat` 格式原始檔案**

MovieLens 10M 的三份資料檔案採用 `::` 作為分隔符：

- `ratings.dat`：格式為 `UserID::MovieID::Rating::Timestamp`，Rating 為 0.5 至 5.0 的半星制浮點數
- `movies.dat`：格式為 `MovieID::Title::Genres`，Genres 以 `|` 分隔
- `tags.dat`：格式為 `UserID::MovieID::Tag::Timestamp`

注意 `movies.dat` 的編碼問題（可能為 ISO-8859-1 而非 UTF-8），需在讀取時指定正確編碼，避免電影標題中的特殊字元（如重音符號）造成解析錯誤。

**任務 1.2.2：完整性驗證**

| 驗證項目 | EDA 預期值 | 容許誤差 |
|---------|-----------|---------|
| 互動總筆數（5-core 前） | 10,000,054 | 0 |
| 互動總筆數（5-core 後） | 9,998,816 | 0 |
| 獨立使用者數 | 69,878 | 0 |
| 獨立電影數（5-core 後） | 10,196 | 0 |
| 半星評分使用比例 | 20.5% | ±0.1% |
| 五星評分比例 | 15.4% | ±0.1% |
| 評分資訊熵 | 1.912 bits | ±0.01 |
| 獨立類型數 | 19 | 0 |
| 標籤事件總數 | 95,580 | 0 |
| 單例標籤比例 | 55.9% | ±0.5% |

**任務 1.2.3：類型向量建構預備**

從 `movies.dat` 萃取所有獨立 Genre 標籤，確認數量為 19 個，建立 Genre → Index 的固定映射表（排除 `(no genres listed)` 的邊緣案例），為後續的 19 維二進位向量做準備。

### 1.3 資料集驗證報告

撰寫一份簡短的驗證記錄（可作為 `README.md` 的一節或獨立的 `data/raw/validation_log.txt`），記錄每個驗證項目的實際讀取值與預期值，作為後續所有步驟的可追溯依據。

**Phase 1 品質閘**：所有驗證項目通過後，方可進入 Phase 2。任何失敗項目需修正讀取邏輯並重新驗證，不得帶著已知的資料不一致性進入後續流程。

**Phase 1 主要輸出物件**：
- 可正常執行的原始資料讀取腳本（整合進 `preprocess.py` 的前段）
- `data/raw/validation_log.txt`：資料完整性驗證記錄

---

## Phase 2：資料前處理、特徵工程與靜態特徵預計算

> **目標**：將原始資料轉換為模型可直接消費的張量（tensor）或序列化物件（pickle），並預先計算所有靜態的混淆變數與成本函數特徵，最終輸出至 `data/processed/`。

### 2.1 共同前處理步驟（`preprocess.py` 主體）

**任務 2.1.1：5-core 過濾**

對 Amazon Beauty 執行嚴格的 5-core 過濾（保留至少 5 筆互動的使用者與物品）。由於 `Beauty_5.json` 可能已為過濾後的版本，需確認是否需要從 `reviews_Beauty.json` 重新執行，或直接採用 `Beauty_5.json` 作為過濾後資料來源。MovieLens 10M 的 5-core 過濾流失率不到 0.02%，直接保留所有記錄。

**任務 2.1.2：使用者與物品 ID 重映射**

將原始的字串型 `reviewerID`（Amazon）/ 整數 `UserID`（MovieLens）與 `asin`（Amazon）/ `MovieID`（MovieLens），分別重映射為從 1 開始的連續整數（0 保留給 PAD token）。建立並儲存雙向映射字典（`user2idx`、`idx2user`、`item2idx`、`idx2item`），這些映射在後續的評估腳本中必須保持一致。

**任務 2.1.3：時序排序與資料切分**

以使用者為單位，依時間戳記升序排列所有互動記錄。採用 leave-one-out 時序切分：

- 測試集：每位使用者的最後一筆互動
- 驗證集：每位使用者的倒數第二筆互動
- 訓練集：其餘所有互動

明確輸出三份分割後的序列字典，格式建議為 `{user_id: [(item_id, rating, timestamp), ...]}`。

**任務 2.1.4：時間間隔計算與對數分桶**

計算每筆互動的相鄰時間間隔 $\Delta t_k = t_k - t_{k-1}$（秒）。對 $\Delta t$ 進行對數空間等頻分桶，分桶數設定：Amazon Beauty 建議 32 桶；MovieLens 建議 64 桶（因 EDA 顯示其 $\Delta t$ 跨度達 8.47 個對數量級，遠大於 Beauty 的 3.26）。需特別處理序列第一筆互動（無前驅時間戳記）的 $\Delta t$ 定義，建議指定為 PAD 桶（如桶號 0）。

**任務 2.1.5：序列截斷**

- Amazon Beauty：最大序列長度 $N_{max} = 50$（覆蓋 P99 = 43），超長序列取最近 50 筆
- MovieLens 10M：最大序列長度 $N_{max} = 200$，超長序列（P99 = 1,057）取最近 200 筆

### 2.2 Amazon Beauty 專屬前處理（`amazon_beauty_loader.py`）

**任務 2.2.1：葉節點類別萃取與 One-hot 查找表建構**

`meta_Beauty.json` 的 `categories` 欄位為巢狀列表（如 `[["Beauty", "Skin Care", "Lotions & Moisturizers"]]`），取最深層（葉節點）類別作為商品的品類代表。建立大小為 222 的葉節點類別 Lookup Table；對出現次數低於門檻（建議 5 次互動）的稀有類別合併為 UNK 類別。

**任務 2.2.2：品牌查找表建構**

建立品牌 Embedding Lookup Table。依 EDA 報告，82.1% 的品牌屬長尾品牌，建議以**10 次互動**為最低門檻，低於此門檻的品牌合併為 UNK（預計大幅縮減有效品牌數，降低嵌入表的過擬合風險）。

**任務 2.2.3：價格特徵計算**

對 `price` 欄位取 $\log_{10}(\text{price})$。缺失的 4.83% 商品以同葉節點類別的中位數 `log_price` 插補；若某類別內無任何有效價格，則退而採用全局中位數（$\log_{10}(11.86) \approx 1.074$）。

**任務 2.2.4：SalesRank 混淆變數 $Z$ 分桶**

萃取 `salesRank` 中 Beauty 類別鍵值對應的排名數值（覆蓋率 88.3%），缺失商品退而採用任意品類的排名（覆蓋率 98.2%）；仍缺失的 1.8% 填入中位數桶（桶號 5）。對 $\log_{10}(\text{salesRank} + 1)$ 進行**10 桶等頻分桶**（EDA 已確認每桶樣本數約 1,068–1,069，均勻性極佳），輸出每個商品對應的桶號 $Z_i \in \{0, 1, \ldots, 9\}$。

**任務 2.2.5：商品元資料查找表的序列化輸出**

將上述四項特徵（葉節點類別 ID、品牌 ID、$\log_{10}(\text{price})$、$Z_i$）彙整為一個以商品 ID 為鍵的 dict，序列化為 `data/processed/amazon_beauty_item_meta.pkl`，供 `amazon_beauty_loader.py` 在訓練時進行 O(1) 查找。

### 2.3 MovieLens 10M 專屬前處理（`movieslens_10M_loader.py`）

**任務 2.3.1：Session 切分（批次補評修正）**

EDA 發現 78.9% 的相鄰評分間隔 $< 60$ 秒，此為「批次補評」行為的人工產物，將污染時序語意。以 **60 秒**為 Session 邊界，對每位使用者的評分序列進行 Session 切分，重新定義 $\Delta t_k$ 為 Session 間的時間差（Session 內部的互動視為同一時刻發生）。此步驟在序列截斷之前執行。

**任務 2.3.2：類型二進位向量建構**

依任務 1.2.3 所建立的 Genre → Index 映射，為每部電影建立 19 維的二進位多熱向量（binary multi-hot vector）。輸出為 `data/processed/ml10m_item_genre.pkl`，格式為 `{movie_id: np.array([0,1,0,...], dtype=np.float32)}`。

**任務 2.3.3：動態混淆變數 $Z_i(t)$ 預計算**

為避免未來資訊洩漏，$Z_i(t)$ 定義為時間點 $t$ 之前物品 $i$ 的累積互動次數（基於訓練集）。由於每筆訓練互動都有其對應的 $Z_i(t)$，採用時間序排序後的掃描（sweep）方式靜態計算，並以稀疏格式（sparse dict 或 CSR 矩陣）儲存，輸出為 `data/processed/ml10m_dynamic_Z.pkl`。

**任務 2.3.4：GenreRed（Jaccard 相似度）預計算**

對每筆訓練互動 $(u, i, t)$，預計算候選電影類型向量與使用者 $t$ 時刻之前觀看歷史的類型 Union 集合之 Jaccard 相似度：

$$\text{GenreRed}(i, \mathcal{H}_{u,<t}) = \frac{|\mathbf{g}_i \cap \mathcal{G}_{u,<t}|}{|\mathbf{g}_i \cup \mathcal{G}_{u,<t}|}$$

由於 Drama 與 Comedy 在 EDA 中被確認具有高頻主導性，同時預計算 **IDF 加權版本**（降低高頻類型權重）以供消融實驗 ML4 使用。輸出格式建議為 `{(user_id, item_id, timestamp): jaccard_score}`，序列化為 `data/processed/ml10m_genre_red.pkl`。

**任務 2.3.5：2009 年資料截斷**

依 EDA 分析，2009 年資料僅含 14,549 筆（截至 1 月 5 日），為避免截斷效應造成的分布偏移，從訓練集中排除 2009 年之後的互動記錄（該部分資料仍可能落入驗證/測試集，保留其評估正確性）。

### 2.4 Tensor 序列化輸出規格

所有前處理結果輸出至 `data/processed/`，建議命名規範如下：

```
data/processed/
├── amazon_beauty/
│   ├── train_seqs.pkl          # {user_id: [(item_id, rating_bin, dt_bin), ...]}
│   ├── val_seqs.pkl            # {user_id: (item_id, rating_bin, dt_bin)}
│   ├── test_seqs.pkl           # {user_id: (item_id, rating_bin, dt_bin)}
│   ├── item_meta.pkl           # {item_id: {cat, brand, log_price, Z}}
│   ├── id_maps.pkl             # {user2idx, idx2user, item2idx, idx2item}
│   └── vocab_sizes.json        # {n_users, n_items, n_cats, n_brands, n_Z_bins, n_dt_bins, n_rating_bins}
└── movielens_10m/
    ├── train_seqs.pkl
    ├── val_seqs.pkl
    ├── test_seqs.pkl
    ├── item_genre.pkl          # {item_id: np.array(19-dim binary)}
    ├── dynamic_Z.pkl           # {(user_id, item_id, ts): Z_value}
    ├── genre_red.pkl           # {(user_id, item_id, ts): jaccard_score}
    ├── genre_red_idf.pkl       # IDF 加權版本
    ├── id_maps.pkl
    └── vocab_sizes.json
```

**Phase 2 品質閘**：對 `train_seqs.pkl` 隨機抽取 100 筆使用者序列，手動核驗時序排序正確性；確認 `val_seqs.pkl` 與 `test_seqs.pkl` 中的物品 ID 均不出現在對應使用者的 `train_seqs.pkl` 中（防止未來資訊洩漏）；確認 `vocab_sizes.json` 中的各維度與實際 Lookup Table 大小一致。

**Phase 2 主要輸出物件**：
- 完整的 `preprocess.py` 腳本（兩資料集的前處理邏輯均封裝其中）
- `data_loaders/amazon_beauty_loader.py`（元資料查找邏輯）
- `data_loaders/movieslens_10M_loader.py`（Genre Jaccard、動態 Z 查找邏輯）
- `data/processed/` 下的所有序列化檔案

---

## Phase 3：工具函式庫與評估基礎設施建置

> **目標**：在模型實作之前，先將損失函數、評估指標、數學工具等基礎設施完整實作並通過單元測試，確保後續所有模型共享同一套評估標準，消除因指標實作不一致導致的比較偏差。

### 3.1 數學工具實作（`utils/math_utils.py`）

**任務 3.1.1：SalesRank 對數分桶函式**

實作 `log_quantile_bucket(values, n_bins)` 函式，對輸入的排名陣列執行 $\log_{10}$ 轉換後的等頻分桶，回傳分桶邊界（bucket edges）與各值對應的桶號。分桶邊界需在前處理完成後固定儲存（fit on training data），在推論時嚴格採用相同邊界，避免資訊洩漏。

**任務 3.1.2：Jaccard 相似度計算函式**

實作 `jaccard_similarity(vec_a, vec_b)` 與批次版本 `batch_jaccard(mat_a, mat_b)`，支援二進位向量輸入。另實作 IDF 加權版本 `weighted_jaccard(vec_a, vec_b, idf_weights)`。需對邊緣情況（兩個向量均為零向量，即電影無類型標記）定義回傳值（建議回傳 0.0，視為零重疊）。

**任務 3.1.3：時間間隔對數分桶函式**

實作 `log_time_bucket(delta_t_seconds, edges)` 函式，將秒為單位的時間間隔映射至對數空間桶號。需處理 $\Delta t \le 0$（同一時刻的連續評分，在 Session 切分後應不再出現，但仍需防禦性處理）。

### 3.2 損失函數實作（`utils/losses.py`）

**任務 3.2.1：標準交叉熵序列損失 $\mathcal{L}_{seq}$**

實作標準的 Next-item prediction 交叉熵損失，作為 IPS 損失的基線對照，也是消融實驗 A2（w/o IPS）的替代損失。

**任務 3.2.2：IPS 加權損失 $\mathcal{L}_{IPS}$**

$$\mathcal{L}_{IPS} = -\sum_{u,i} \frac{y_{ui}}{p_i} \log \hat{y}_{ui}$$

分別實作兩種變異數縮減策略：

- **Clipped IPS**：`w = min(1/p_i, tau)`，閾值 $\tau$ 由 `configs/` 設定，Amazon Beauty 預設 30，MovieLens 預設 100
- **Self-Normalized IPS**：`w = (1/p_i) / sum(1/p_j for all j in batch)`，以 batch 內的分母做歸一化

兩者以統一介面封裝，透過設定參數切換，供消融實驗 A4 使用。

**任務 3.2.3：對抗損失 $\mathcal{L}_{adv}$**

$$\mathcal{L}_{adv} = -\text{CE}(D(h_t), Z)$$

對抗損失為標準多分類交叉熵（$Z$ 為分桶後的離散類別），但需注意：此損失在**反向傳播至序列編碼器時**，梯度方向由 GRL 反轉（最大化鑑別器的預測誤差）；在反向傳播至**鑑別器本身**時，梯度方向正常（最小化鑑別器的預測誤差）。GRL 的實作必須在 `causal_deconfounder.py` 中完成，`losses.py` 僅計算原始 CE 值。

**任務 3.2.4：效用選擇損失 $\mathcal{L}_C$**

$$\mathcal{L}_C = -\log P(i_t \mid h_t) = -\log \frac{\exp(U(u, i_t, t))}{\sum_{j \in \mathcal{I}} \exp(U(u, j, t))}$$

注意全物品 softmax（Full Softmax）在物品數較大時（Amazon Beauty: 12,101；MovieLens: 10,196）計算成本是可接受的（相較於語言模型的 vocabulary），無需 Sampled Softmax，但需確認 batch 計算的正確向量化形式。

**任務 3.2.5：聯合目標函數整合**

$$\mathcal{L}_{total} = \mathcal{L}_{IPS} + \lambda_{adv}\mathcal{L}_{adv} + \mathcal{L}_C + \lambda_{reg}\|\theta\|^2$$

實作 `CombinedLoss` 類別，接受各損失的 scalar 值與權重係數，回傳加權總損失。$\lambda_{adv}$ 與 $\lambda_{reg}$ 從 `configs/` 讀取。

### 3.3 評估指標實作（`utils/metrics.py`）

**任務 3.3.1：Full Ranking 評估協定**

實作在**全物品排序**（Full Ranking）而非隨機負採樣下的指標計算。對每位測試使用者，計算模型對所有物品的分數，排除已在訓練集中出現過的物品，在剩餘物品中對目標物品進行排名。

**任務 3.3.2：主要指標函式**

- `ndcg_at_k(ranks, k)` → NDCG@K，K ∈ {5, 10, 20}
- `hr_at_k(ranks, k)` → HR@K，K ∈ {5, 10, 20}
- `mrr(ranks)` → MRR

所有指標均支援 batch 計算並回傳 per-user 值（便於後續分組分析），最後對全使用者取平均。

**任務 3.3.3：分組評估指標（長尾/頭部分析）**

依物品全局互動頻率，建立 Head（> P80）/ Torso（P20–P80）/ Tail（< P20）三組標籤，存為 `data/processed/*/item_popularity_group.pkl`。在 `metrics.py` 中實作 `group_metrics(ranks, item_ids, group_labels)` 函式，分別回傳三組的 NDCG@10 與 HR@10，用以驗證假說 H2a。

**任務 3.3.4：去混淆品質指標**

實作 `confounding_auc(h_states, Z_labels)` 函式：以已訓練的潛在狀態 $h_t$ 作為輸入特徵，訓練一個 Logistic Regression 分類器（固定 5-fold CV），預測混淆變數桶號 $Z$，回傳多分類 AUC 值。此指標用於驗證假說 H2b：若 AUC 趨近於隨機基線，代表 $h_t \perp Z$ 的去混淆目標已達成。

### 3.4 單元測試

**任務 3.4.1**：對 `losses.py` 的每個損失函式，以已知輸入與預期輸出撰寫至少 3 個單元測試案例，特別針對邊緣情況（如 $p_i \to 0$ 的 IPS 截斷行為、全 0 logits 的 softmax 行為）。

**任務 3.4.2**：對 `metrics.py` 的每個指標，以小型 toy example（如 3 個使用者 × 5 個物品）手算預期值後驗證輸出正確性。

**Phase 3 品質閘**：所有單元測試通過（zero failures）；`losses.py` 的各損失函式在簡單 toy model 上能正確反向傳播（梯度不為 NaN/Inf）。

**Phase 3 主要輸出物件**：
- `utils/math_utils.py`（含單元測試）
- `utils/losses.py`（含單元測試）
- `utils/metrics.py`（含單元測試）
- `data/processed/*/item_popularity_group.pkl`（物品流行度分組標籤）

---

## Phase 4：CEINN 三大模組實作

> **目標**：依照研究方法 §3.1、§3.2、§3.3 的數學規格，完整實作 CEINN 的三個核心模組，並以前向傳播（forward pass）正確性作為各模組的驗收標準。

### 4.1 模組 3.1：Sequential Backbone（`models/sequential_backbone.py`）

**任務 4.1.1：嵌入層建構**

實作三種嵌入的聯合建構：

$$e_k = E_i(i_k) + E_r(r_k) + E_t(\Delta t_k)$$

- `ItemEmbedding`：大小為 `(n_items + 1) × d`，+1 為 PAD token（index 0）
- `RatingEmbedding`：Amazon Beauty 為 5 個整數等級（1–5），MovieLens 為 10 個半星等級（0.5–5.0，映射至 1–10），大小為 `(n_rating_bins + 1) × d`
- `TemporalEmbedding`：大小為 `(n_dt_bins + 1) × d`，+1 為 PAD

所有嵌入使用 Xavier 均勻初始化，PAD index 的嵌入向量固定為零向量（`padding_idx=0`）。

**任務 4.1.2：Causal-Masked Self-Attention 實作**

以 PyTorch 的 `nn.MultiheadAttention` 為基礎，建構因果遮罩矩陣：

$$M_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

實作多層 Transformer Encoder 堆疊（層數 $L$ 由 config 控制），每層包含：Multi-head Self-Attention → Add & Norm → Feed-Forward → Add & Norm。注意 PyTorch 的 `attn_mask` 格式（加法 mask，而非布林 mask），需確認傳入的矩陣型別正確。

**任務 4.1.3：潛在狀態 $h_t$ 輸出**

取 Transformer 最後一層在位置 $t$（序列最後一個有效 token 位置）的輸出向量作為 $h_t$，維度為 $d$。此向量將分別傳入模組 3.2（去混淆）與模組 3.3（效用計算），**不做任何額外的語意特徵融合**，嚴守特徵隔離原則。

**任務 4.1.4：前向傳播測試**

以 batch_size = 4、序列長度 = 10、$d = 64$ 為參數，確認：
- 輸出形狀為 `(batch_size, d)` ✓
- 設定因果遮罩後，位置 $t$ 的輸出不受位置 $t+1$ 以後輸入的影響（可透過修改輸入序列後半段並確認前半段輸出不變來驗證）✓

### 4.2 模組 3.2：Causal Deconfounder（`models/causal_deconfounder.py`）

**任務 4.2.1：傾向估計器（Propensity Estimator）實作**

實作一個淺層 MLP（2 層，隱藏維度建議為 64），輸入為混淆變數桶號 $Z_i$ 的 Embedding，輸出為 Softmax 機率（代表物品落入各流行度桶的估計機率）。

$$p_i = P(E_i = 1 \mid Z_i)$$

傾向分數以 Sigmoid 輸出形式實作（二元曝光模型），或以 Softmax 形式實作（多類別流行度分桶）；論文中需明確說明採用哪種形式。建議採用 Sigmoid 形式，以「物品是否被曝光給使用者」為二元預測目標，訓練標籤設定為：出現在訓練集中的互動對 $(u, i)$ 為正例，其餘為負例（以均勻隨機負採樣）。

**任務 4.2.2：梯度反轉層（GRL）實作**

實作自定義的 `GradientReversalLayer`，繼承 `torch.autograd.Function`：

```python
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None
```

$\alpha$ 採用漸進式調度：

$$\alpha(p) = \frac{2}{1 + \exp(-10p)} - 1, \quad p = \frac{\text{current epoch}}{\text{total epochs}}$$

在訓練初期（$p \to 0$），$\alpha \to 0$，GRL 幾乎不反轉梯度，讓序列編碼器先學習基本的推薦能力；隨訓練深入，$\alpha \to 1$，逐漸強化去混淆壓力。

**任務 4.2.3：鑑別器（Discriminator）實作**

實作一個 2 層 MLP 鑑別器，輸入為 $\text{GRL}(h_t)$，輸出為對 $K = 10$ 個混淆變數桶的分類 logit：

$$D: \mathbb{R}^d \to \mathbb{R}^{K}$$

**任務 4.2.4：前向傳播測試**

確認：在反向傳播時，$h_t$ 關於 $\mathcal{L}_{adv}$ 的梯度方向與鑑別器參數的梯度方向相反（GRL 作用驗證）。

### 4.3 模組 3.3：Economics Utility Module（`models/economics_utility.py`）

**任務 4.3.1：Amazon Beauty 顯性成本函數實作**

$$C(i, t) = \alpha_1 \log(\text{price}_i) + \alpha_2 \phi(\text{category}_i) + \alpha_3 \psi(\text{brand}_i) + \alpha_4 \eta(\text{salesRank}_i) + \alpha_5 \big(\text{price}_i \times \phi(\text{category}_i)\big)$$

實作要點：
- $\phi(\text{category}_i)$：品類 Embedding（One-hot Lookup Table，維度 222，取自 `item_meta.pkl`），映射到向量後再線性投影為純量（或直接以加權求和的方式計算）
- $\psi(\text{brand}_i)$：品牌 Embedding，同上處理
- 交互作用項 $\alpha_5$：$\log(\text{price}_i)$ 純量 × 品類 Embedding 線性投影純量，逐元素相乘後求和
- 係數 $\alpha_1, \ldots, \alpha_5$ 皆為可學習參數（`nn.Parameter`），以均勻分布 $[0, 0.1]$ 初始化

**任務 4.3.2：MovieLens 隱性行為成本函數實作**

$$C(i, t) = \beta_1 \cdot \text{GenreRed}(i, \mathcal{H}_{u,<t}) + \beta_2 \cdot \text{RecencyPress}(t) + \beta_3 \cdot \text{PopPress}(i)$$

實作要點：
- $\text{GenreRed}$：從預計算的 `genre_red.pkl` 直接查找（O(1)）
- $\text{RecencyPress}(t)$：定義為目前序列時間點 $t$ 相對於使用者最早互動時間的歸一化值（反映累積觀影疲勞）
- $\text{PopPress}(i)$：動態 $Z_i(t)$ 的歸一化版本
- 係數 $\beta_1, \beta_2, \beta_3$ 均為可學習參數，初始化為較小正值（如 0.01），並監控其學習動態（依 EDA 的 GenreRed 飽和假說警示）

**任務 4.3.3：主觀價值 $V$ 與效用 $U$ 計算**

$$V(u, i, t) = f_\theta(h_t, E_i(i))$$

以雙線性映射（Bilinear mapping）作為 $f_\theta$ 的基本形式：

$$V(u, i, t) = h_t^\top W_V E_i(i)$$

或以 MLP 形式：$f_\theta = \text{MLP}(\text{concat}(h_t, E_i(i)))$。論文中需確認採用哪種形式，建議以雙線性作為預設（計算效率高，且更符合傳統 CF 的計算範式）。

最終效用：$U(u, i, t) = V(u, i, t) - \lambda_u C(i, t)$

**任務 4.3.4：使用者成本敏感度 $\lambda_u$ 網路**

$$\lambda_u = \sigma(w^\top z_u), \quad z_u = \frac{1}{|\mathcal{H}_u|}\sum_{k} E_i(i_k)$$

$z_u$ 以使用者歷史交互物品 ID 嵌入的平均（mean pooling）計算，為避免退化（$\lambda_u \to 0$），在訓練中監控其分布；若持續趨近於零，加入 $[-\log \lambda_u]$ 的正則化項強制其遠離零點。

### 4.4 主模型整合（`models/ceinn.py`）

**任務 4.4.1：三模組串接**

`CEINNModel` 類別封裝三個子模組，`forward()` 方法的流程為：

```
輸入序列 (item_ids, rating_ids, dt_ids)
        ↓
Sequential Backbone → h_t                    (§3.1)
        ├──→ Causal Deconfounder              (§3.2)
        │       ├── GRL(h_t) → Discriminator → L_adv
        │       └── Propensity Estimator → p_i
        └──→ Economics Utility Module         (§3.3)
                ├── Value: V(u,i,t) = f(h_t, E_i)
                ├── Cost: C(i,t) from meta features
                ├── Lambda: λ_u = σ(w⊤z_u)
                └── Utility: U = V - λ_u·C → L_C
```

**任務 4.4.2：前向傳播整合測試**

以 toy batch（4 個使用者 × 序列長度 10 × 候選物品 50）確認：完整前向傳播不出現 NaN/Inf；三個損失值均為有限正數；反向傳播後所有可學習參數的梯度均非零。

**Phase 4 品質閘**：前向傳播整合測試通過；$\text{GRL}$ 梯度反轉驗證通過；因果遮罩未來資訊隔離驗證通過。

**Phase 4 主要輸出物件**：
- `models/sequential_backbone.py`
- `models/causal_deconfounder.py`
- `models/economics_utility.py`
- `models/ceinn.py`
- `models/__init__.py`

---

## Phase 5：基線模型實作與評估

> **目標**：在統一的資料格式與評估協定下，訓練所有 8 個基線模型，建立可信賴的性能基準線，並確認評估框架的正確性（以 PopRec 的 NDCG 值作為最低下界校驗）。

### 5.1 基線模型實作策略

實驗設計文件所列的 8 個基線模型，依工程複雜度分為兩類：

**可調用現有開源實作的基線**（建議以 recbole 框架或單獨 repo 整合）：
- PopRec（非個人化流行度推薦，自行實作即可，5 行程式碼以內）
- BPR-MF（矩陣分解，RecBole 或自行實作）
- GRU4Rec（RecBole 或官方實作）
- SASRec（建議直接沿用 CEINN Sequential Backbone 的程式碼，移除去混淆與效用模組）
- BERT4Rec（RecBole 或官方實作）

**需自行實作的因果推薦基線**：
- IPS-SASRec：SASRec + `utils/losses.py` 的 IPS 加權損失（直接組合）
- DICE：參考原論文實作，核心為解耦合 Interest/Conformity 嵌入
- CauseRec：參考原論文實作，核心為反事實序列增強

### 5.2 訓練配置對齊

所有基線模型採用以下統一設定（與 CEINN 完全一致）：

| 配置項 | Amazon Beauty | MovieLens 10M |
|--------|--------------|----------------|
| 嵌入維度 | 64 | 128 |
| 序列截斷長度 | 50 | 200 |
| 最大訓練 epoch | 200 | 200 |
| Early stopping patience | 10（基於 Val NDCG@10） | 10 |
| 優化器 | Adam（lr = 5e-4） | Adam（lr = 5e-4） |
| Batch size | 256 | 512 |
| 評估協定 | Full Ranking | Full Ranking |

### 5.3 基線模型評估結果記錄

訓練完成後，以 5 個隨機種子各跑一次，記錄每個基線在驗證集與測試集的指標：

- NDCG@{5, 10, 20}
- HR@{5, 10, 20}
- MRR

同時記錄：長尾（Tail）/ 中段（Torso）/ 頭部（Head）子集的 NDCG@10，作為 CEINN 去混淆效果的比較基準。

**Phase 5 品質閘**：PopRec 在 Amazon Beauty 上的 NDCG@10 應遠低於 SASRec（預期差距 > 5%），否則代表評估協定存在問題；SASRec 的訓練損失曲線應正常收斂（無發散或平台期）。

**Phase 5 主要輸出物件**：
- 各基線模型的程式碼（整合至 `models/` 或獨立 `baselines/` 子目錄）
- `results/baselines_summary.csv`：基線模型指標彙總表
- 每個基線的最佳模型權重（`.pt` 檔）

---

## Phase 6：CEINN 主模型訓練流程實作與超參數搜索

> **目標**：完整實作 `train.py` 與 `evaluate.py`，執行兩階段超參數搜索，確定最終的模型配置，並完成 CEINN-Full 在兩個資料集上的最終訓練與評估。

### 6.1 訓練主腳本（`train.py`）

**任務 6.1.1：聯合優化目標的反向傳播設計**

聯合損失的反向傳播涉及兩個競爭目標（推薦任務 vs. 對抗去混淆），需特別設計梯度流：

- **推薦損失**（$\mathcal{L}_{IPS} + \mathcal{L}_C$）：正常反向傳播至所有參數
- **對抗損失**（$\lambda_{adv} \mathcal{L}_{adv}$）：透過 GRL 反轉梯度，只更新序列編碼器（使其學會"欺騙"鑑別器）；同時正常更新鑑別器本身

建議採用**兩步更新策略**：

```
Step 1：凍結鑑別器，以 L_IPS + L_C + λ_adv * L_adv（反轉梯度）更新編碼器與效用模組
Step 2：凍結編碼器，以 L_adv（正常梯度）更新鑑別器
```

**任務 6.1.2：GRL 調度整合**

在每個 epoch 開始時，依 $\alpha(p) = \frac{2}{1+\exp(-10p)}-1$ 更新 GRL 的 $\alpha$ 值，$p = \text{epoch}/\text{total\_epochs}$。

**任務 6.1.3：訓練監控與日誌**

每個 epoch 後記錄：$\mathcal{L}_{IPS}$、$\mathcal{L}_{adv}$、$\mathcal{L}_C$、$\mathcal{L}_{total}$ 的訓練集數值；Val NDCG@10；鑑別器對混淆變數 $Z$ 的分類準確率（反映去混淆進展）。使用 TensorBoard 或 CSV 日誌記錄，方便後續的訓練曲線可視化。

**任務 6.1.4：Early stopping 與模型存儲**

以 **Val NDCG@10 為 Early stopping 準則**，patience = 10 epochs。在每個驗證集最優 epoch 儲存完整模型狀態（`state_dict`）至 `checkpoints/`。

### 6.2 評估腳本（`evaluate.py`）

**任務 6.2.1：Full Ranking 評估實作**

對每位測試使用者：
1. 計算模型對所有物品的效用分數 $U(u, i, t)$
2. 排除訓練集中已出現的物品（seen items masking）
3. 對剩餘物品按分數降序排列，取目標物品的排名
4. 計算 NDCG@{5,10,20}、HR@{5,10,20}、MRR

**任務 6.2.2：分析性指標計算**

在評估腳本中加入：
- 各物品流行度組（Head / Torso / Tail）的分組指標
- 收集所有測試使用者的 $h_t$ 向量，以供去混淆品質分析（`confounding_auc` 計算）
- 收集 $\lambda_u$ 值分布

### 6.3 超參數搜索（Two-Stage Strategy）

**任務 6.3.1：Stage 1 粗搜（Random Search）**

以 `random.seed(42)` 為起點，從以下空間均勻採樣 **20 組超參數組合**：

```yaml
d:          [32, 64, 128]
L:          [1, 2, 3]
H:          [1, 2, 4]
dropout:    [0.1, 0.2, 0.3, 0.5]
lr:         [1e-4, 5e-4, 1e-3]
lambda_adv: [0.001, 0.01, 0.1, 1.0]
lambda_reg: [1e-5, 1e-4, 1e-3]
tau:        Amazon: [10, 20, 50]; ML: [50, 100, 200]
ips_mode:   [clipped, self_normalized]
```

每組超參數僅訓練 **50 epochs**（不使用 Early stopping），以 Val NDCG@10 排序，取 **Top-3** 候選配置。

**任務 6.3.2：Stage 2 精搜（Full Training）**

對 Top-3 配置以完整 200 epochs + Early stopping 重新訓練，選取驗證集最優配置作為最終設定，更新至 `configs/amazon_beauty.yaml` 與 `configs/movielens_10M.yaml`。

**任務 6.3.3：$\lambda_{adv}$ 敏感性分析**

固定其他超參數，系統性地掃描 $\lambda_{adv} \in \{0.001, 0.01, 0.1, 0.5, 1.0\}$，繪製 Val NDCG@10 vs. $\lambda_{adv}$ 的敏感性曲線，以及鑑別器準確率 vs. $\lambda_{adv}$ 的曲線，揭示推薦準確性與去混淆強度的 Pareto 折衷關係（此圖為論文重要分析圖之一）。

### 6.4 CEINN-Full 最終訓練

以 Stage 2 確定的最優超參數，以 **5 個不同隨機種子** 分別完整訓練 CEINN-Full，每次記錄最佳測試集指標（Best epoch 基於驗證集選擇，測試集僅評估一次）。計算跨種子的均值與標準差，作為論文的主要結果。

**Phase 6 品質閘**：CEINN-Full 的 Val NDCG@10 訓練曲線收斂（無發散）；訓練過程中鑑別器準確率從高位逐漸下降至接近隨機基線（驗證 GRL 有效運作）；最終測試集結果優於 SASRec（至少在 NDCG@10 上）。

**Phase 6 主要輸出物件**：
- `train.py`（完整訓練腳本）
- `evaluate.py`（完整評估腳本）
- `configs/amazon_beauty.yaml`（最終超參數）
- `configs/movielens_10M.yaml`（最終超參數）
- `checkpoints/`：CEINN-Full 5 個種子的最佳權重
- `results/ceinn_full_main_results.csv`：主要結果表格

---

## Phase 7：消融實驗與假說驗證

> **目標**：依照實驗設計文件 Section 5 的消融矩陣，系統性地移除或替換各模組，精確歸因 CEINN 每個設計決策的貢獻；同時執行資料集專屬消融，驗證各項數據驅動的設計選擇。

### 7.1 通用消融實驗（A1–A8）

依照以下矩陣逐一訓練消融變體，每個變體以 **3 個隨機種子** 執行，報告均值 ± 標準差：

| 變體 | 核心修改 | 對應假說 | 預期方向 |
|------|---------|---------|---------|
| A1: w/o Rating Emb | 移除 $E_r$，僅使用 $E_i + E_t$ | H1 | 性能微降（評分嵌入提供情感強度信號）|
| A2: w/o IPS | 以標準 CE 損失取代 $\mathcal{L}_{IPS}$ | H2a | 長尾子集 NDCG@10 顯著下降 |
| A3: w/o GRL | 移除對抗去混淆，$h_t$ 不經 GRL | H2b | 鑑別器 AUC 上升（$h_t$ 重新編碼流行度信息）|
| A4: Clipped vs. SN-IPS | 切換 IPS 變異數縮減策略 | H2c | MovieLens 上，Clipped 訓練曲線方差較大 |
| A5: w/o Economics | $U$ 退化為 dot-product score | H3 | 性能下降，尤以成本相關場景明顯 |
| A6: w/o Cost | $U = V$，移除 $C$ 項 | H3 | 性能微降 |
| A7: fixed $\lambda_u$ | $\lambda_u$ 固定為常數 0.5 | H3a | 性能下降，個人化成本敏感度有效 |
| A8: w/o Temporal Enc | 移除 $E_t$，僅使用 $E_i + E_r$ | H1 | 性能微降（MovieLens 下降幅度應大於 Beauty，因 ML 序列更長）|

### 7.2 Amazon Beauty 專屬消融（AB1–AB3）

**AB1：Static Z（以滾動頻率取代 salesRank）**

將混淆變數 $Z_i$ 從 salesRank 分桶改為全局累積互動頻率的等頻分桶。預期：salesRank 提供比頻率更乾淨的流行度信號，AB1 應略低於完整模型。若差距不顯著，需在論文中討論兩者代理能力的差異。

**AB2：移除 Price × Category 交互作用項**

成本函數退化為：$C(i,t) = \alpha_1 \log(\text{price}_i) + \alpha_2 \phi(\text{category}_i) + \alpha_3 \psi(\text{brand}_i) + \alpha_4 \eta(\text{salesRank}_i)$（移除 $\alpha_5$ 交互作用項）。依 EDA 的 F 代理統計量 = 57.07，預期此項對高端品類（如 Lotions、Skincare）的性能貢獻顯著。

**AB3：移除品牌嵌入**

評估品牌作為成本信號的邊際效益。依 EDA，品牌覆蓋率 82.8%，長尾品牌佔 82.1%，預期貢獻有限，但值得量化。

### 7.3 MovieLens 10M 專屬消融（ML1–ML4）

**ML1：Static Z**

以全局累積頻率取代動態 $Z_i(t)$。依 EDA 的流行度排名波動率 23.65%，預期動態 $Z$ 對 ML 資料集的貢獻顯著（對應假說 H2a 的動態混淆變數版本）。

**ML2：不做 Session 切分**

直接使用原始 $\Delta t$（含 78.9% 的 < 1 分鐘間隔）訓練。此消融量化批次補評行為對時序建模的污染程度——若 ML2 性能下降顯著，代表 Session 切分是不可缺少的前處理步驟，應在論文的前處理節中明確說明。

**ML3：移除 GenreRed 項**

$C(i,t)$ 退化為 $\beta_2 \cdot \text{RecencyPress}(t) + \beta_3 \cdot \text{PopPress}(i)$。依 EDA 的警示（飽和假說未獲直接確認），此消融的結果具有重要學術誠信意義：
- 若 ML3 性能下降：GenreRed 有效，EDA 的飽和假說後顧之憂可消除
- 若 ML3 性能不降反升：需在論文中誠實討論 §3.3.4 的假說局限性，並提出修正方向

**ML4：IDF 加權 Jaccard vs. 標準 Jaccard**

比較 GenreRed 的兩種計算方式，驗證 Drama/Comedy 降權對 Genre 疲勞建模準確性的影響。

### 7.4 分析性實驗

**B1：$\lambda_u$ 分布可視化**

對 CEINN-Full 訓練完成後，收集所有使用者的 $\lambda_u$ 值，繪製分布直方圖。需觀察：分布是否非退化（存在明顯的跨使用者差異）；是否存在可解釋的使用者群組（如 $\lambda_u$ 高的使用者對應低互動頻率的使用者，暗示對新鮮感的高敏感度）。

**B2：成本模組對物品排序的影響分析**

比較 CEINN-Full 與 A5（w/o Economics）對同一組測試使用者的推薦列表差異：

- 在 Amazon Beauty 中：高價物品在加入成本模組後，是否在對成本敏感使用者（$\lambda_u$ 高）的推薦列表中排名下降？
- 在 MovieLens 中：高 GenreRed 物品（與使用者近期觀影高度重疊的電影）是否在加入成本模組後被「多樣化推薦」所取代？

此分析提供模型行為的可解釋性論據，是論文 Qualitative Analysis 節的重要素材。

**Phase 7 品質閘**：A2、A3、A5 三個核心消融的性能下降幅度，在 Wilcoxon 檢定下均顯著（$p < 0.05$），驗證三大模組均有實質貢獻；否則需回頭審視對應模組的實作是否存在 bug。

**Phase 7 主要輸出物件**：
- `results/ablation_results.csv`：所有消融變體的完整指標表
- `results/lambda_u_distribution.png`（B1 分析圖）
- `results/cost_ranking_shift.csv`（B2 分析資料）
- 更新後的模型程式碼（若消融實驗發現任何 bug）

---

## Phase 8：實驗結果彙整、統計分析與學術討論

> **目標**：對所有實驗結果進行系統性的統計分析，撰寫論文所需的結果表格、圖表與討論文字，並針對任何未達預期的結果給出學術上誠實且深度的解釋。

### 8.1 主要結果表格製作

**任務 8.1.1：統一主結果表**

製作論文的主要比較表，格式如下（以 Amazon Beauty 為例，MovieLens 同格式）：

| Model | NDCG@5 | NDCG@10 | NDCG@20 | HR@5 | HR@10 | HR@20 | MRR |
|-------|--------|---------|---------|------|-------|-------|-----|
| PopRec | - | - | - | - | - | - | - |
| BPR-MF | - | - | - | - | - | - | - |
| GRU4Rec | - | - | - | - | - | - | - |
| SASRec | - | - | - | - | - | - | - |
| BERT4Rec | - | - | - | - | - | - | - |
| IPS-SASRec | - | - | - | - | - | - | - |
| DICE | - | - | - | - | - | - | - |
| CauseRec | - | - | - | - | - | - | - |
| **CEINN** | **-** | **-** | **-** | **-** | **-** | **-** | **-** |
| *Improve* | *%* | *%* | *%* | *%* | *%* | *%* | *%* |

所有結果以均值 ± 標準差呈現（跨 5 個種子），以 Wilcoxon 顯著性符號（`*`/`**`）標記。

**任務 8.1.2：長尾分組結果表**

製作 Head / Torso / Tail 三組的 NDCG@10 比較表，著重展示 CEINN 相對 SASRec 在長尾組的差距，以量化去混淆對稀疏物品的改善效益。

**任務 8.1.3：消融實驗結果表**

製作 A1–A8 通用消融與資料集專屬消融的完整比較表，每行對應一個消融變體，以 CEINN-Full 為 100% 基線，報告各項指標的相對變化百分比。

### 8.2 統計顯著性分析

**任務 8.2.1：Wilcoxon Signed-Rank Test**

對主要結果表中的每個基線模型，以 CEINN-Full vs. 該基線在 5 個種子上的 NDCG@10 配對值執行 Wilcoxon 單邊檢定（H₁：CEINN > Baseline），報告 $p$-value。對消融實驗（A2、A3、A5 為最重要的三個），同樣執行顯著性檢定，基於 3 個種子的配對值。

**任務 8.2.2：效果量計算**

除 $p$-value 外，計算 CEINN vs. 最強基線的 Cohen's $d$ 效果量，提供除統計顯著性之外的實際效益量化。

### 8.3 訓練動態可視化

**任務 8.3.1：訓練曲線圖**

繪製 CEINN-Full 在兩個資料集上的訓練曲線，包含：
- $\mathcal{L}_{total}$、$\mathcal{L}_{IPS}$、$\mathcal{L}_{adv}$、$\mathcal{L}_C$ 的 per-epoch 曲線（訓練集）
- Val NDCG@10 曲線（含 Early stopping 標記）
- 鑑別器對 $Z$ 的分類準確率曲線（應隨 GRL 訓練逐漸下降至隨機基線附近）

**任務 8.3.2：$\lambda_{adv}$ 敏感性曲線圖**

繪製 Val NDCG@10 vs. $\lambda_{adv}$（5 個候選值），以及鑑別器準確率 vs. $\lambda_{adv}$，揭示 Pareto 折衷關係。

**任務 8.3.3：去混淆品質圖（$h_t$ vs. $Z$ AUC 曲線）**

繪製訓練過程中，$h_t$ 對混淆變數 $Z$ 的預測 AUC 隨 epoch 的變化曲線（CEINN-Full vs. A3 w/o GRL 的對照），量化對抗去混淆的漸進效果。

### 8.4 學術討論撰寫

**任務 8.4.1：主要假說的討論**

對 H1、H2、H3 三個假說，依實驗結果逐一評估：
- 假說是否獲支持？支持的程度（全部、部分、否定）？
- 量化支持的關鍵數字（如「IPS 加權使長尾 NDCG@10 提升 X.X%」）
- 兩個資料集的結果是否一致？若不一致，原因為何？

**任務 8.4.2：EDA 預警問題的誠實討論**

針對 EDA 報告中已預警的潛在問題，在實驗結果對照後進行誠實討論：

- **MovieLens 的 GenreRed 假說（最重要）**：若 ML3 消融顯示 GenreRed 無效或負效，需深度討論批次補評行為對 Jaccard 信號的語意污染；提出後續改進方向（如以 Session 內類型重複度取代跨時間 Jaccard）。

- **Amazon Beauty 的評分通貨膨脹**：57.7% 的五星評分是否導致 $E_r$ 的其他評分等級嵌入學習不足？從 A1（w/o Rating Emb）的消融結果可窺見，若 A1 性能幾乎不降，代表評分嵌入的信號確實被通貨膨脹稀釋。

- **MovieLens 的 IPS 爆炸問題**：對比 Amazon Beauty（最大 $1/p_i = 86.2$）與 MovieLens（最大 $1/p_i = 6,972.8$），在結果表格中觀察兩者 IPS 校正的效益差異，並討論極端逆傾向分數對模型性能的實際影響。

**任務 8.4.3：方法局限性（Limitations）撰寫**

誠實陳述 CEINN 的以下局限性：

1. **冷啟動問題**：序列長度極短的使用者（Amazon Beauty P50 = 6）對 IPS 傾向估計與潛在狀態 $h_t$ 的估計精度均有限制，CEINN 在此場景下的優勢受限。

2. **成本函數的觀測假設**：Amazon Beauty 的顯性成本函數依賴 price 等元資料的正確性（缺失率 4.83%），MovieLens 的 GenreRed 依賴 Session 切分的設定（60 秒邊界為任意設定），這些均為可質疑的設計假設。

3. **靜態混淆變數的限制（Amazon Beauty）**：salesRank 為靜態特徵，無法捕捉流行度的時間演變（如季節性爆款），相較於 MovieLens 的動態 $Z_i(t)$，在時序因果推論上有理論上的不完整性。

4. **可擴展性**：Full Ranking 評估在物品數較多的工業場景（百萬量級）中計算不可行；CEINN 的聯合訓練目標（三個損失函數）的訓練時間約為 SASRec 的 3–5 倍，需討論其工業部署的可行性。

**任務 8.4.4：未來研究方向撰寫**

基於上述局限性，提出以下可延伸的研究方向：

- 將靜態 salesRank 分桶升級為動態流行度軌跡建模（如 Hawkes Process）
- 在 Session 層面重新定義時序偏好演進，解決 MovieLens 的批次補評問題
- 探索多目標優化框架（Pareto optimization）以更系統地處理推薦準確性與公平性的折衷
- 將 CEINN 框架延伸至多模態推薦（圖像、文字），探索語意特徵與因果推論的整合邊界

### 8.5 最終可重現性封存

**任務 8.5.1：程式碼整理與 README 撰寫**

完整撰寫 `README.md`，包含：環境配置指令（`requirements.txt` 對應的 `conda`/`pip` 命令）、一鍵前處理指令、一鍵訓練指令（以 `configs/` 驅動）、一鍵評估指令，以及預期的輸出數值範圍（供讀者核驗環境正確性）。

**任務 8.5.2：隨機種子清單公開**

在 `README.md` 或附錄中公開所有實驗的 5 個隨機種子列表，確保任何人在相同環境下可以完整重現論文結果。

**任務 8.5.3：最終 configs 凍結**

將 Stage 2 超參數搜索確定的最終超參數完整寫入 `configs/amazon_beauty.yaml` 與 `configs/movielens_10M.yaml`，設定所有欄位（含消融變體的開關旗標），確保 config 驅動的實驗可以一鍵重現任何消融變體。

**Phase 8 品質閘**：論文的所有表格數值均可從 `results/` 目錄下的 CSV 直接追溯；至少一位組員在全新環境（乾淨 conda 環境）下按 README 指令成功重現主要結果（NDCG@10 誤差在 ±0.2% 以內）。

**Phase 8 主要輸出物件**：
- `results/main_comparison_table.csv`
- `results/ablation_results.csv`
- `results/group_metrics.csv`（Head/Torso/Tail 分組）
- `results/significance_tests.csv`（Wilcoxon 檢定結果）
- `results/figures/`（所有訓練曲線與分析圖表）
- 完整且可執行的 `README.md`
- 凍結的 `configs/amazon_beauty.yaml` 與 `configs/movielens_10M.yaml`

---

## 附錄：各階段依存關係與關鍵路徑

```
Phase 1（資料驗證）
    │
    ▼
Phase 2（前處理）──────────────────────────────┐
    │                                            │
    ▼                                            ▼
Phase 3（工具函式庫）                     [特徵工程輸出]
    │                                            │
    ├──────────────────────────────┐             │
    ▼                              ▼             ▼
Phase 4（CEINN 模型實作）    Phase 5（基線模型）
    │                              │
    └──────────┬────────────────────┘
               ▼
         Phase 6（CEINN 訓練與超參數搜索）
               │
               ▼
         Phase 7（消融實驗）
               │
               ▼
         Phase 8（結果分析與討論）
```

**關鍵路徑**：Phase 1 → 2 → 3 → 4 → 6 → 7 → 8（總計 8 個階段，Phase 5 基線模型實作可與 Phase 4 的 CEINN 模型實作並行進行，是縮短整體執行時間的主要並行化機會）。

---

## 附錄：風險登記與應變措施

| 風險 | 發生階段 | 嚴重程度 | 應變措施 |
|------|---------|---------|---------|
| GRL 對抗訓練不收斂 | Phase 6 | 高 | 降低 $\lambda_{adv}$，延長 $\alpha$ warm-up；鑑別器學習率設為編碼器的 5 倍 |
| MovieLens IPS 梯度爆炸 | Phase 6 | 高 | 優先啟用 Self-Normalized IPS；加入梯度裁剪（max norm = 1.0） |
| GenreRed 消融（ML3）負效益 | Phase 7 | 中 | 誠實在論文中討論；提出 Session 層級的 GenreRed 改進方向 |
| $\lambda_u$ 退化至 0 | Phase 6/7 | 中 | 加入 $[-\log \lambda_u]$ 正則化；對 $\lambda_u$ 設定下界 Clamp（0.01）|
| 基線模型（DICE/CauseRec）無開源實作可用 | Phase 5 | 低-中 | 以 IPS-SASRec 的強化版（更深 MLP）作為替代比較基線；在論文 Related Work 中說明 |
| Amazon Beauty 的 salesRank 缺失率高於預期 | Phase 2 | 低 | 以互動頻率排名作為後備混淆變數；報告兩種 $Z$ 的比較結果（即 AB1 消融）|

---

*本執行計畫根據 `ceinn_research.md`（研究方法）、`CEINN_Experiment_Design.md`（實驗設計）、`EDA_Report_CEINN_Amazon_Beauty.md`、`EDA_Report_CEINN_MovieLens_10M.md` 綜合撰寫，所有設計決策均可回溯至對應的統計依據或研究假說。*
