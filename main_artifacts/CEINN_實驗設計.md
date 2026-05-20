# CEINN 實驗設計文件

**研究題目**：CEINN: Causal Economics-Informed Neural Networks for Sequential Recommendation  
**文件類型**：論文實驗設計（Experimental Design）  

---

## 目錄

1. [研究問題與假說體系](#1-研究問題與假說體系)
2. [資料集與前處理規格](#2-資料集與前處理規格)
3. [評估指標定義](#3-評估指標定義)
4. [基線模型選定與理由](#4-基線模型選定與理由)
5. [消融實驗設計](#5-消融實驗設計)
6. [超參數設定與搜索策略](#6-超參數設定與搜索策略)
7. [統計顯著性與可重現性規範](#7-統計顯著性與可重現性規範)
8. [跨資料集比較的特殊考量](#8-跨資料集比較的特殊考量)
9. [預期結果與可能的失敗模式](#9-預期結果與可能的失敗模式)
10. [實驗進度里程碑](#10-實驗進度里程碑)

---

## 1. 研究問題與假說體系

### 1.1 核心研究問題

本研究的核心問題可被形式化為三個層次遞進的假說：

**H1（序列基礎假說）**：純 ID 驅動、具備因果遮罩的序列編碼器，能夠在不引入任何外部語意特徵的前提下，估計出一個資訊量充足的使用者潛在狀態 $h_t$，作為後續因果推論的乾淨基底。

**H2（因果去混淆假說）**：在 H1 的基礎上，透過 IPS 加權損失與對抗性正則化的雙重機制，可以顯著削弱潛在狀態 $h_t$ 對流行度偏差的依賴，從而在長尾物品（cold/rare items）的推薦準確率上，相較未去混淆的模型取得實質提升。

**H3（經濟學效用假說）**：在 H2 的基礎上，將推薦決策建模為效用最大化過程（$U = V - \lambda_u C$），相較於直接以匹配分數（dot-product score）進行排序，能夠捕捉使用者異質性的成本敏感度，從而在偏好與成本高度相關的場景（Amazon Beauty 的高價品類；MovieLens 的類型飽和情境）上取得額外的性能增益。

### 1.2 可操作化的子假說

| 編號 | 假說描述 | 對應消融實驗 | 驗證指標 |
|------|---------|------------|---------|
| H2a | IPS 加權比標準 CE 損失在長尾物品上表現更優 | Ablation A2 | 長尾子集的 NDCG@10 |
| H2b | GRL 對抗去混淆使 $h_t \perp Z$ | Ablation A3 | $h_t$ 對 $Z$ 的預測 AUC（越低越好） |
| H2c | Self-Normalized IPS 的訓練穩定性優於 Clipped IPS（MovieLens）| Ablation A4 | 訓練損失曲線方差 |
| H3a | 使用者成本敏感度 $\lambda_u$ 在不同使用者間存在顯著差異 | 分析性實驗 B1 | $\lambda_u$ 的分布統計 |
| H3b | 高價格/高類型重疊物品在加入成本模組後排序下降 | 分析性實驗 B2 | 排序前後之成本分布差異 |

---

## 2. 資料集與前處理規格

### 2.1 資料集概況對比

| 特性 | Amazon Beauty | MovieLens 10M |
|------|--------------|----------------|
| 互動筆數 | 198,502 | 9,998,816 |
| 使用者數 | 22,363 | 69,878 |
| 物品數 | 12,101 | 10,196 |
| 互動矩陣密度 | 0.073% | 1.403% |
| 流行度 Gini | 0.499（中等） | 0.796（高度集中）|
| 序列長度 P50 | 6 | 69 |
| 最大逆傾向分數 | 86.2 | 6,972.8 |

兩資料集的特性差異具有互補的設計價值：Amazon Beauty 驗證明確經濟成本函數的有效性，MovieLens 10M 驗證隱性行為摩擦力建模，且後者的流行度偏差更為嚴峻，為 IPS 機制的壓力測試提供了更苛刻的環境。

### 2.2 資料前處理流程

#### 2.2.1 共同步驟（兩資料集均適用）

**Step 1：5-core 過濾**  
保留至少有 5 筆互動的使用者與物品，以確保序列編碼器有足夠的學習信號。Amazon Beauty 需要執行此步驟；MovieLens 10M 原始資料已近似滿足（流失率 < 0.02%）。

**Step 2：時序排序**  
以使用者為單位，依時間戳記（unixReviewTime / timestamp）升序排列互動記錄，確保因果遮罩機制的語意正確性。

**Step 3：訓練/驗證/測試切分**  
採用 **leave-one-out 時序切分（Temporal leave-one-out split）**：
- 測試集：每位使用者的**最後一筆**互動
- 驗證集：每位使用者的**倒數第二筆**互動
- 訓練集：其餘所有互動

此策略優於隨機切分，原因在於它嚴格模擬了真實推薦場景——模型只能觀測歷史，對未來保持不可知。

**Step 4：序列截斷**  
- Amazon Beauty：最大序列長度設為 **50**（覆蓋 P99 = 43），超長序列取最近 50 筆
- MovieLens 10M：最大序列長度設為 **200**，對超長序列採用滑動窗口取最近紀錄；P99 = 1,057 的極端超長序列以此截斷

**Step 5：ID 重映射**  
將 item ID 與 user ID 重映射為連續整數（0-indexed），並為 PAD（填充）、MASK（遮蔽）、UNK（未知）保留特殊 token。

#### 2.2.2 Amazon Beauty 專屬步驟

**Step 6a：混淆變數 $Z$ 的建構**  
依 §3.2.4，對 salesRank 進行對數轉換後等頻分桶：

$$Z_i = \text{Bucket}_{10}(\log_{10}(\text{salesRank}_i + 1))$$

分桶數設為 10（EDA 已確認等頻分桶後每桶約 1,068–1,069 個樣本）。SalesRank 缺失的 1.8% 物品填入中位數桶（Bucket 5）。

**Step 7a：成本函數特徵準備**  
- `price`：取 $\log_{10}(\text{price}_i)$，缺失值（4.83%）以同品類中位數插補
- `category`：取葉節點類別，建立大小為 222 的 Embedding Lookup Table；低頻類別（互動數 < 5）合併為 UNK
- `brand`：建立大小為 2,077 的 Lookup Table；互動數 < 10 的長尾品牌合併為 UNK（約 82% 的長尾品牌需合併，建議門檻為 10 次互動）
- `salesRank`：取 $\log_{10}(\text{salesRank}_i + 1)$，同上缺失值處理

#### 2.2.3 MovieLens 10M 專屬步驟

**Step 6b：Session 切分（批次補評修正）**  
鑑於 EDA 發現 78.9% 的相鄰互動發生在 1 分鐘內，直接使用原始時間戳記將導致時序語意污染。建議將連續時間間隔 $\Delta t < 60$ 秒的評分事件合併為同一 session，並以 session 為單位重新定義時間間隔 $\Delta t_{session}$（session 間的時間差），以還原更接近真實偏好演進的時序信號。

**Step 7b：動態混淆變數 $Z_i(t)$ 的計算**  
為避免未來資訊洩漏，$Z_i(t)$ 採用滾動計算：對每筆訓練互動 $(u, i, t)$，$Z_i(t)$ 定義為時間點 $t$ 之前物品 $i$ 的累積互動次數，預先於 `preprocess.py` 中計算並存為靜態查找表。

**Step 8b：類型特徵準備**  
- `genres`：取 19 維二進位向量（multi-hot encoding）
- `GenreRed`：預先計算每筆訓練互動 $(u, i, t)$ 所對應的 Jaccard 相似度：$\text{GenreRed}(i, \mathcal{H}_{u,<t}) = \frac{|\mathbf{g}_i \cap \mathcal{G}_{u,<t}|}{|\mathbf{g}_i \cup \mathcal{G}_{u,<t}|}$，其中 $\mathcal{G}_{u,<t}$ 為使用者 $t$ 時刻前觀看過之所有類型集合
- **IDF 加權建議**：考量 Drama 與 Comedy 的高頻主導性（EDA Section 4.1），建議在 Jaccard 計算中引入 IDF 加權，降低高頻類型的影響力

**Step 9b：資料截斷**  
2009 年僅含 14,549 筆互動（截至 1 月 5 日），建議從訓練集中排除以避免截斷效應造成的分布偏移。

---

## 3. 評估指標定義

### 3.1 主要評估指標

採用序列推薦領域的標準離線評估協定：**全物品排序（Full Ranking）**，即對所有未出現在訓練集中的物品進行排序，不採用隨機負採樣（以確保評估的無偏性）。

**NDCG@K（Normalized Discounted Cumulative Gain）**

$$\text{NDCG@K} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{\text{DCG@K}(u)}{\text{IDCG@K}(u)}$$

本研究報告 K = 5、10、20 三個層次。主要指標為 **NDCG@10**。

**HR@K（Hit Ratio）**

$$\text{HR@K} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \mathbf{1}[\text{target item} \in \text{Top-K}]$$

本研究報告 K = 5、10、20 三個層次。

**MRR（Mean Reciprocal Rank）**

$$\text{MRR} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{\text{rank}_u}$$

MRR 對排名靠前的精確性更敏感，有助於捕捉模型在頭部推薦上的品質。

### 3.2 輔助分析指標

除整體指標外，本研究設計以下**分組評估**以驗證特定假說：

**長尾物品子集指標（驗證 H2a）**  
將測試集物品依全局互動頻率分為三組：
- **頭部**（Head）：互動數 > P80 全局百分位
- **中段**（Torso）：P20 ~ P80
- **長尾**（Tail）：互動數 < P20

分別報告三組的 NDCG@10 與 HR@10，以量化 IPS 去混淆對長尾物品的實質改善。

**表示層去混淆指標（驗證 H2b）**  
訓練一個輕量分類器（Logistic Regression）以 $h_t$ 預測混淆變數桶 $Z$，報告其 AUC。若對抗去混淆有效，此 AUC 應趨近於隨機基線（0.50 + 1/K，其中 K 為桶數）。

**成本敏感度分布（驗證 H3a）**  
報告學得之使用者成本敏感度 $\lambda_u = \sigma(w^\top z_u)$ 的全局分布（均值、標準差、四分位距）。

### 3.3 評估協定補充說明

- **驗證集用途**：超參數選擇（Early stopping 的基準）與模型選擇，不得用於最終比較
- **測試集嚴格封存**：所有基線模型與消融變體在最終超參數確定後，僅報告一次測試集結果
- **冷啟動使用者的處理**：Amazon Beauty 測試集中序列長度 $\le 3$ 的使用者（對應 P50 = 6）單獨報告，以揭示短序列場景下各模型的極限

---

## 4. 基線模型選定與理由

### 4.1 基線模型列表

| 模型 | 類別 | 主要特性 | 選定理由 |
|------|------|---------|---------|
| **PopRec** | Non-personalized | 全局流行度排序 | 最低基線；量化流行度偏差的天花板效應 |
| **BPR-MF** | Collaborative Filtering | 矩陣分解 + BPR 損失 | 代表傳統 CF，不具備序列建模能力 |
| **GRU4Rec** | Sequential（RNN-based） | GRU 序列編碼 | 早期序列推薦代表；測試 Transformer 相對 RNN 的優勢 |
| **SASRec** | Sequential（Transformer） | 因果遮罩自注意力 | CEINN 模組 3.1 的直接基礎；去除因果與效用後的 ablation 上界 |
| **BERT4Rec** | Sequential（Transformer） | 雙向遮罩語言模型 | 代表雙向注意力路線，與 CEINN 的單向因果機制形成對照 |
| **IPS-SASRec** | Debiased Sequential | SASRec + IPS 加權 | 驗證 CEINN 的對抗去混淆（GRL）相對於單純 IPS 的增量貢獻 |
| **DICE** | Causal Sequential | 因果解耦合嵌入 | 代表現有因果推薦模型；直接競爭對手 |
| **CauseRec** | Causal Sequential | 反事實資料增強 | 驗證 IPS 策略相對反事實增強策略的效益 |

### 4.2 基線模型的對齊原則

為確保比較的公平性，所有模型採用以下統一設定：

- 嵌入維度 $d$：對齊 CEINN 的最優設定（建議 64 或 128）
- 序列截斷長度：與 CEINN 相同（Amazon Beauty: 50；MovieLens: 200）
- 訓練/驗證/測試切分：完全相同的資料分割
- 負採樣策略：Full Ranking 評估（不使用負採樣）
- 最大訓練 epoch：設定共同上限（如 200 epochs），以 Validation NDCG@10 為 Early stopping 基準，patience = 10

### 4.3 CEINN 相對各基線的預期增量

```
PopRec
  └─(+序列建模)→ GRU4Rec / SASRec
                  └─(+因果 IPS)→ IPS-SASRec
                                  └─(+對抗 GRL)→ CEINN w/o Economics
                                                  └─(+效用模組)→ CEINN (Full)
```

此層次結構使每個增量的貢獻可被精確歸因，與消融實驗設計（Section 5）形成呼應。

---

## 5. 消融實驗設計

消融實驗是驗證 CEINN 各模組貢獻的核心工具。設計原則：**每次僅移除或替換一個模組**，保持其餘部分不變。

### 5.1 模組消融矩陣

| 變體名稱 | 序列編碼 (§3.1) | IPS 加權 (§3.2) | GRL 對抗 (§3.2) | 效用模組 (§3.3) | 備注 |
|---------|:--------------:|:--------------:|:--------------:|:--------------:|------|
| **CEINN-Full** | ✓ | ✓ | ✓ | ✓ | 完整模型 |
| **A1: w/o Rating Emb** | ✓（無 $E_r$） | ✓ | ✓ | ✓ | 驗證評分嵌入的貢獻 |
| **A2: w/o IPS** | ✓ | ✗（用 CE 取代） | ✓ | ✓ | 驗證 IPS 加權的因果校正效果 |
| **A3: w/o GRL** | ✓ | ✓ | ✗ | ✓ | 驗證對抗去混淆在表示空間上的作用 |
| **A4: Clipped IPS vs. SN-IPS** | ✓ | Clipped | ✓ | ✓ | 比較兩種變異數縮減策略 |
| **A5: w/o Economics** | ✓ | ✓ | ✓ | ✗（用 dot-product 取代） | 驗證效用模組的淨貢獻 |
| **A6: w/o Cost** | ✓ | ✓ | ✓ | ✓（$U = V$，移除 $C$） | 驗證成本項的獨立貢獻 |
| **A7: fixed $\lambda_u$** | ✓ | ✓ | ✓ | ✓（$\lambda_u$ 固定為常數） | 驗證個人化成本敏感度的增益 |
| **A8: w/o Temporal Enc** | ✓（無 $E_t$） | ✓ | ✓ | ✓ | 驗證時間編碼的貢獻 |

### 5.2 資料集專屬消融

#### Amazon Beauty 專屬

| 變體名稱 | 說明 | 驗證假說 |
|---------|------|---------|
| **AB1: Static Z（raw freq）** | 以靜態互動頻率取代 salesRank 作為 $Z$ | 確認 salesRank 相較頻率作為混淆變數代理的優越性 |
| **AB2: w/o Price×Category** | 成本函數移除交互作用項 $\alpha_5$ | 驗證品類-價格交互作用項（F = 57.07）的貢獻 |
| **AB3: w/o Brand Emb** | 成本函數移除品牌嵌入 $\psi$ | 評估品牌特徵的邊際效益 |

#### MovieLens 10M 專屬

| 變體名稱 | 說明 | 驗證假說 |
|---------|------|---------|
| **ML1: Static Z** | 以全局累積頻率取代動態 $Z_i(t)$ | 驗證動態混淆變數的必要性（對應流行度波動率 23.65%）|
| **ML2: w/o Session Split** | 不進行 Session 切分，直接使用原始 $\Delta t$ | 量化批次補評行為對時序建模的污染程度 |
| **ML3: w/o GenreRed** | 成本函數移除 $\beta_1 \cdot \text{GenreRed}$ 項 | 評估類型冗餘項在批次補評混淆下的實際貢獻 |
| **ML4: IDF-weighted Jaccard** | GenreRed 計算引入 IDF 加權 | 驗證高頻類型（Drama/Comedy）降權對準確率的影響 |

### 5.3 聯合目標函數的超參數敏感性分析

最終優化目標

$$\mathcal{L} = \mathcal{L}_{IPS} + \lambda_{adv} \mathcal{L}_{adv} + \lambda_{reg} \|\theta\|^2$$

中的 $\lambda_{adv}$ 對去混淆強度有關鍵影響。設計網格搜索（Grid Search）：

- $\lambda_{adv} \in \{0.001, 0.01, 0.1, 0.5, 1.0\}$
- $\lambda_{reg} \in \{1 \times 10^{-5}, 1 \times 10^{-4}, 1 \times 10^{-3}\}$

報告 Validation NDCG@10 對 $\lambda_{adv}$ 的敏感性曲線，以揭示去混淆與推薦準確性之間的 Pareto 折衷關係。

---

## 6. 超參數設定與搜索策略

### 6.1 模型架構超參數

| 超參數 | 搜索範圍 | Amazon Beauty 建議初始值 | MovieLens 10M 建議初始值 |
|--------|---------|------------------------|------------------------|
| 嵌入維度 $d$ | {32, 64, 128} | 64 | 128 |
| Transformer 層數 $L$ | {1, 2, 3} | 2 | 2 |
| 注意力頭數 $H$ | {1, 2, 4} | 2 | 4 |
| Dropout rate | {0.1, 0.2, 0.3, 0.5} | 0.2 | 0.1 |
| 序列截斷長度 $N_{max}$ | — | 50 | 200 |
| 評分嵌入分桶數（$E_r$） | — | 5（整數評分） | 10（半星制） |
| 時間間隔分桶數（$E_t$） | {16, 32, 64} | 32 | 64 |
| 傾向估計器層數 | {1, 2}（Shallow MLP） | 2 | 2 |
| GRL 的 $\alpha$ 調度 | 線性 warm-up | $\alpha: 0 \to 1$，前 10 epochs | 同左 |

### 6.2 訓練超參數

| 超參數 | 搜索範圍 | 說明 |
|--------|---------|------|
| 學習率 | {1e-4, 5e-4, 1e-3} | Adam 優化器 |
| Batch size | {128, 256, 512} | Amazon Beauty 建議 256；MovieLens 建議 512 |
| Maximum epochs | 200 | |
| Early stopping patience | 10 | 基於 Validation NDCG@10 |
| IPS 截斷閾值 $\tau$ | {10, 20, 50}（AB）；{50, 100, 200}（ML） | 兩資料集閾值範圍截然不同，見 EDA 分析 |
| $\lambda_{adv}$ | {0.001, 0.01, 0.1, 1.0} | 對抗損失權重 |
| $\lambda_{reg}$ | {1e-5, 1e-4, 1e-3} | L2 正則化 |

### 6.3 GRL 的 $\alpha$ 調度策略

梯度反轉層的強度 $\alpha$ 建議採用漸進式調度，避免訓練初期的大幅梯度反轉破壞序列編碼器的表示學習：

$$\alpha(p) = \frac{2}{1 + \exp(-10p)} - 1, \quad p = \frac{\text{current epoch}}{\text{total epochs}} \in [0, 1]$$

此調度使 $\alpha$ 從 0 平滑增長至接近 1，與 DANN（Domain-Adversarial Neural Networks）的原始設計一致。

### 6.4 超參數搜索策略

考量到訓練資源限制，建議採用 **Two-stage 搜索策略**：

**Stage 1（粗搜）**：以 CEINN-Full 模型在驗證集上進行 Random Search，固定訓練 50 epochs（不使用 Early stopping），搜索 20 組組合，記錄 Top-3 配置。

**Stage 2（精搜）**：對 Stage 1 的 Top-3 配置以完整 200 epochs + Early stopping 重新訓練，選取最優配置後，以此固定所有消融實驗的架構超參數，僅調整與消融相關的目標函數超參數。

---

## 7. 統計顯著性與可重現性規範

### 7.1 多次運行與隨機種子

所有主要結果（CEINN-Full 與 8 個基線模型）須以 **5 個不同隨機種子** 分別訓練，報告：

- 均值 $\pm$ 標準差
- 最優單次結果（Best run）

消融實驗因組合數量多，可採用 3 個隨機種子。

### 7.2 顯著性檢定

採用 **Wilcoxon signed-rank test**（成對非參數檢定），以 CEINN-Full 與各基線模型的跨種子 NDCG@10 為觀測值，報告 $p$-value。顯著性門檻設為 $p < 0.05$，論文中以 `*` 標記，$p < 0.01$ 以 `**` 標記。

選用 Wilcoxon 而非 t-test 的理由：推薦指標的分布通常非常態，且樣本量（5個種子）較小，非參數方法更為穩健。

### 7.3 可重現性規範

- 所有實驗的隨機種子列表將於論文附錄或 README 中公開
- 前處理後的資料（`data/processed/`）及最佳模型權重將透過版本化方式歸檔
- 設定文件（`configs/amazon_beauty.yaml`、`configs/movielens_10M.yaml`）記錄最終採用的所有超參數，確保一鍵可重現

---

## 8. 跨資料集比較的特殊考量

### 8.1 兩資料集的差異化設計選擇對比

由於 CEINN 針對兩個資料集採用了不同的混淆變數建構策略（靜態 salesRank 分桶 vs. 動態滾動頻率）與不同的成本函數形式（顯性金錢成本 vs. 隱性行為摩擦力），**跨資料集的直接性能數值比較並不是本研究的主要目標**。兩資料集的實驗設計旨在驗證 CEINN 框架的**通用性（generalizability）**，即相同的框架結構能否在兩種截然不同的領域中，均帶來相對對應基線的提升。

### 8.2 領域特定的關鍵觀察點

**Amazon Beauty 的關鍵觀察**：

1. 效用模組中的 $\alpha_5(\text{price} \times \phi(\text{category}))$ 交互作用項，是否在不同品類的測試子集上帶來差異化的提升（例如：高價精華液品類的推薦準確率提升是否大於低價指甲油品類）？

2. IPS 截斷閾值 $\tau$ 的選擇（建議 20–50）在此資料集中的敏感性，相較 MovieLens（建議 50–200）是否更為和緩？

**MovieLens 10M 的關鍵觀察**：

1. Session 切分前後，模型的 NDCG@10 差異幅度，量化批次補評行為對時序建模的實際污染程度（對應 ML2 消融實驗）。

2. 動態 $Z_i(t)$ vs. 靜態 $Z$ 的比較（ML1 消融），以流行度波動率 23.65% 作為預期效益的先驗估計。

3. GenreRed 項的有效性在 EDA 中未獲直接支持（評分方向與飽和假說相反）。若 ML3 消融（移除 GenreRed）後性能不降反升，則需在論文中誠實討論 §3.3.4 的假說局限性，並提出後續修正方向（如以互動序列多樣性取代類型重疊度）。

### 8.3 兩資料集結果的共同報告格式

為使審稿人易於比較，建議論文中以**統一表格**呈現兩個資料集的所有主要結果，列明各基線與 CEINN-Full 的 NDCG@{5, 10, 20} 及 HR@{5, 10, 20}，並以顯著性符號標記，最後一列報告 CEINN-Full 相對最強基線的相對提升百分比。

---

## 9. 預期結果與可能的失敗模式

### 9.1 預期的正面結果

| 場景 | 預期觀察 | 理論依據 |
|------|---------|---------|
| CEINN-Full vs. SASRec | NDCG@10 提升 3–8% | 因果去混淆修正曝光偏差，效用模組捕捉決策結構 |
| 長尾子集 vs. 頭部子集的提升差異 | 長尾提升幅度更大 | IPS 加權對冷門物品的補償機制 |
| MovieLens IPS 截斷敏感性 | 極端截斷（$\tau \ll 100$）損害性能 | 過度截斷削弱了對嚴重長尾偏差的修正能力 |
| $\lambda_u$ 分布 | 非退化，存在跨使用者差異 | 經濟學理論預測消費者異質性 |

### 9.2 可能的失敗模式與應對方案

**失敗模式 1：GRL 對抗訓練不收斂（鑑別器 vs. 編碼器出現模式崩潰）**  
信號：訓練損失 $\mathcal{L}_{adv}$ 在初期後停止下降，$h_t$ 對 $Z$ 的預測 AUC 未能收斂至隨機基線附近。  
應對：降低 $\lambda_{adv}$，延長 GRL 的 $\alpha$ warm-up 週期；嘗試將鑑別器的學習率設為編碼器的 5–10 倍（加速去混淆端的學習）。

**失敗模式 2：MovieLens 的 IPS 梯度仍不穩定（即使截斷）**  
信號：訓練損失出現劇烈震盪，甚至 NaN。  
應對：優先啟用 Self-Normalized IPS（而非 Clipped IPS）；若問題持續，可暫時降低 IPS 加權的學習率（使用 GradNorm 或梯度裁剪）。

**失敗模式 3：效用模組的 $V - \lambda_u C$ 分解退化（$\lambda_u \to 0$ 或 $C \to 0$）**  
信號：$\lambda_u$ 的學習值全部趨近於零，或成本函數的係數 $\alpha_i / \beta_i$ 在訓練後趨近於零。  
應對：對 $\lambda_u$ 加入範圍約束（如 Clamp 至 $[0.01, 1.0]$）；對成本函數係數加入 L1 正則化以避免稀疏退化。

**失敗模式 4：MovieLens GenreRed 項負向貢獻**  
信號：ML3 消融（移除 GenreRed）後 NDCG@10 不降反升，幅度超過 0.5%。  
應對：誠實在論文 §3.3.4 中補充討論，指出批次補評行為導致 Jaccard 信號的語意混淆（EDA 已預警此風險）；提出以 session 內類型重複度取代全局 Jaccard 的改進方向。

---

## 10. 實驗進度里程碑

| 里程碑 | 主要工作 | 輸出物件 |
|--------|---------|---------|
| **M1：資料前處理完成** | 執行 `preprocess.py`，生成 Amazon Beauty 與 MovieLens 的 processed tensors；確認 5-core、序列切分、$Z$ 分桶均正確 | `data/processed/` 下的 pickle/tensor 檔案 |
| **M2：模組單元測試** | 分別測試 `sequential_backbone.py`、`causal_deconfounder.py`、`economics_utility.py` 的前向傳播，確認 loss 收斂方向正確 | 測試腳本 log |
| **M3：基線模型訓練** | 訓練 8 個基線模型（含 SASRec 作為 CEINN 的直接前身），記錄 Validation NDCG@10 曲線 | 基線結果表格 |
| **M4：CEINN-Full 初始訓練** | 以預設超參數訓練 CEINN-Full，觀察訓練穩定性與收斂行為（特別關注 IPS 爆炸與 GRL 收斂） | 訓練曲線圖 |
| **M5：超參數搜索** | 執行 Two-stage 超參數搜索，確定最終配置 | 更新 `configs/*.yaml` |
| **M6：消融實驗** | 依 Section 5 的矩陣，訓練 A1–A8 與資料集專屬消融，各 3 個種子 | 消融結果表格 |
| **M7：最終評估與顯著性檢定** | 以 5 個種子重新訓練所有主要模型，執行 Wilcoxon 檢定，生成分組（長尾/頭部）子集評估結果 | 論文主結果表格 |
| **M8：分析性實驗** | 可視化 $\lambda_u$ 分布（B1）；分析成本模組對物品排序的影響（B2）；繪製 $h_t$ 對 $Z$ 預測的 AUC 學習曲線 | 論文分析圖表 |

---

## 附錄 A：聯合目標函數的統一說明

CEINN 完整的訓練目標函數為：

$$\mathcal{L}_{total} = \underbrace{\mathcal{L}_{IPS}}_{\text{去偏序列損失}} + \underbrace{\lambda_{adv} \mathcal{L}_{adv}}_{\text{表示去混淆}} + \underbrace{\mathcal{L}_C}_{\text{效用選擇損失}} + \underbrace{\lambda_{reg}\|\theta\|^2}_{\text{L2 正則化}}$$

其中：

$$\mathcal{L}_{IPS} = -\sum_{u,i} \frac{y_{ui}}{p_i} \log \hat{y}_{ui}, \qquad p_i = P(E_i = 1 \mid Z_i)$$

$$\mathcal{L}_{adv} = -\text{CE}(D(\text{StopGrad}(\text{GRL}(h_t))), Z)$$

$$\mathcal{L}_C = -\log P(i_t \mid h_t) = -\log \frac{\exp(U(u, i_t, t))}{\sum_{j} \exp(U(u, j, t))}$$

注意：$\mathcal{L}_{IPS}$ 與 $\mathcal{L}_C$ 在監督信號上存在部分重疊——兩者皆針對已觀測互動 $(u, i, t)$ 進行最大似然估計，但前者在樣本權重上引入了 IPS 校正。實作時可選擇以 $\mathcal{L}_C$ 取代 $\mathcal{L}_{IPS}$ 作為主損失（即效用模組的 NLL 損失直接以 IPS 加權），或保留兩者並分別以不同學習率優化。論文中需明確說明實際採用的組合方式。

---

## 附錄 B：程式碼模組與實驗對應關係

| 實驗設計元素 | 對應程式碼模組 |
|------------|--------------|
| 序列編碼 / $h_t$ 估計 | `models/sequential_backbone.py` |
| 傾向估計器 / IPS 加權 | `models/causal_deconfounder.py` + `utils/losses.py` |
| GRL 對抗去混淆 | `models/causal_deconfounder.py`（Discriminator + GRL）|
| 效用模組 $V$, $C$, $\lambda_u$ | `models/economics_utility.py` |
| 截斷/自平滑 IPS 演算法 | `utils/losses.py` |
| 評估指標計算 | `utils/metrics.py` |
| salesRank 對數分桶 | `utils/math_utils.py` |
| Jaccard 相似度 | `utils/math_utils.py` |
| Amazon Beauty 特徵載入 | `data_loaders/amazon_beauty_loader.py` |
| MovieLens 特徵載入 | `data_loaders/movieslens_10M_loader.py` |
| 模型訓練主循環 | `train.py` |
| 測試集評估 | `evaluate.py` |
| 超參數設定 | `configs/amazon_beauty.yaml`、`configs/movielens_10M.yaml` |

---

*本文件根據 `ceinn_research.md`、`EDA_Report_CEINN_Amazon_Beauty.md`、`EDA_Report_CEINN_MovieLens_10M.md` 及對應的專案目錄結構綜合撰寫。所有實驗設計決策均力求回溯至研究假說，並以 EDA 實證數據為依據。*
