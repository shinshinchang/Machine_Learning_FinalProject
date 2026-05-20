**基於因果經濟學感知神經網路的序列推薦系統**

CEINN: Causal Economics-Informed Neural Networks for Sequential Recommendation

**專題成員**：113405208 黃旭寬 傳院二丁、113703018 黃育緯 資訊二、113306004 常興宥 資管二乙、113306009 張綺均 資管二甲、114306085 吳敘辰 資管一甲

**指導老師**：魏綾音 助理教授  
---

### **3.1 Sequential Backbone and Latent State Estimation**

本節將闡述 CEINN 框架的基礎模組。不同於傳統序列推薦模型旨在最大化下一個物品的預測機率，本模組的統計任務定位於部分可觀測隨機過程（Partially Observed Stochastic Process）中的潛在狀態估計（Latent State Estimation）。

#### **3.1.1 Problem Formulation from a Statistical Learning View**

給定使用者的歷史互動軌跡，本模組旨在建立一個非線性映射函數 $\\mathcal{F}\_{\\theta}$，從觀測到的序列資料中萃取出時間 $t$ 的潛在決策狀態（Latent decision state）$h\_t$：

$$h\_t \= \\mathcal{F}\_{\\theta}(i\_1, r\_1, t\_1, \\dots, i\_n, r\_n, t\_n)$$

在因果推論的視角下，$h\_t$ 應被視為使用者在特定時間點的動態偏好表示（Dynamic preference representation），而非單純的歷史特徵聚合。

#### **3.1.2 The Principle of Representation Disentanglement**

在訓練強大的序列編碼器時，機器學習模型極易傾向吸收資料中的捷徑特徵（Shortcut features）。若在特徵萃取階段過早引入外部特徵，潛在表示 $h\_t$ 將退化為多重因素的糾纏態：

$$h\_t \= f(\\text{true preference}, \\text{exposure mechanism}, \\text{market bias})$$

例如，若直接將 Amazon Beauty 中的價格（price）或銷售排名（salesRank）特徵輸入序列模型，$h\_t$ 將無可避免地吸收市場的流行度偏差（Popularity bias）與曝光機制（Exposure mechanism）。這會導致後續的因果模組完全無法進行解耦合（Disentangle）。因此，本模組嚴格遵循特徵隔離原則，確保編碼器僅處理互動本體（Interaction identity）、序列順序（Interaction order）、時間演進（Temporal evolution）與顯性反饋強度（Explicit feedback intensity），完全摒棄外部語意的介入。

#### **3.1.3 Mathematical Formulation of Interaction Sequences**

定義使用者 $u$ 在時間 $t$ 之前的歷史觀測集合為 $\\mathcal{H}\_u$：

$$\\mathcal{H}\_u \= \\{ (i\_1, r\_1, t\_1), (i\_2, r\_2, t\_2), \\dots, (i\_n, r\_n, t\_n) \\}$$

其中 $i\_k$ 表示互動的商品（如 Amazon asin 或 MovieLens MovieID），$r\_k$ 為偏好強度（Amazon overall 為 $1 \\sim 5$整數；MovieLens Rating 為 $0.5 \\sim 5.0$ 浮點數），$t\_k$ 為時間戳記（unixReviewTime 或 Timestamp）。

為了將上述異質資料映射至連續向量空間，我們為每個互動步驟建構聯合嵌入（Joint Embedding）$e\_k$：

$$e\_k \= E\_i(i\_k) \+ E\_r(r\_k) \+ E\_t(\\Delta t\_k)$$

* **Item Embedding ($E\_i$)**: $E\_i(i\_k) \\in \\mathbb{R}^d$ 僅捕捉商品唯一識別碼（Item ID），刻意排除 description 或 genres 等外部語意擴充特徵。**在本研究的架構中，此純粹的 ID 嵌入向量將直接作為後續所有模組（包含因果去混淆與經濟學效用評估）中商品的唯一表徵。**  
* **Rating Embedding ($E\_r$)**: 由於 $(Watch, rating=5) \\neq (Watch, rating=1)$，$E\_r(r\_k)$ 將使用者的顯性偏好強度直接注入序列表示中。  
* **Temporal Encoding ($E\_t$)**: 考慮到使用者的偏好演進非獨立同分配（Non-i.i.d.），我們定義時間間隔 $\\Delta t\_k \= t\_k \- t\_{k-1}$，並對其進行連續時間編碼或對數分桶（Logarithmic bucketization），以捕捉時間衰減效應。

#### **3.1.4 Causal-Masked Sequential State Estimation**

為了處理長距離依賴（Long-range dependency）並捕捉多重興趣，我們採用具備因果遮罩（Causal mask）的自注意力機制（Self-Attention）作為核心架構（類似 SASRec）。給定輸入矩陣 $X \= \[e\_1, e\_2, \\dots, e\_n\]$，透過線性投影取得 $Q, K, V$：

$$Q=XW\_Q, \\quad K=XW\_K, \\quad V=XW\_V$$

為了嚴格遵守時間的單向性並避免未來資訊洩漏（Future leakage），我們引入遮罩矩陣 $M$：

$$M\_{ij} \= \\begin{cases} 0 & \\text{if } j \\le i \\\\ \-\\infty & \\text{if } j \> i \\end{cases}$$

注意力權重的計算如下：

$$\\text{Attention}(Q, K, V) \= \\text{softmax}\\left( \\frac{QK^T}{\\sqrt{d}} \+ M \\right) V$$

#### **3.1.5 Statistical Significance of the User-State**

經過 Transformer Encoder 的多層堆疊後，我們取得時間點 $t$ 的輸出 $h\_t$：

$$h\_t \= \\text{TransformerEncoder}(\\mathcal{H}\_{u, \<t})$$

在統計機率的框架下，$h\_t$ 不應被單純視為靜態的使用者嵌入（User embedding），而是條件潛在狀態估計（Conditional latent state estimate）。它逼近了神經網路版本的貝氏濾波（Bayesian filtering）：

$$h\_t \\approx P(\\text{user preference state at time } t \\mid \\mathcal{H}\_{u, \<t})$$

#### **3.1.6 Observational Training Objective**

在觀測資料的層次上，我們使用 Next-item prediction 作為基礎訓練目標。給定當前狀態 $h\_t$，模型預測下一個商品 $i\_{t+1}$ 的機率分佈：

$$P(i\_{t+1} \\mid h\_t) \= \\text{softmax}(W h\_t)$$

並透過最小化交叉熵損失函數（Cross-entropy loss）進行最佳化：

$$\\mathcal{L}\_{seq} \= \- \\sum \\log P(i\_{t+1} \\mid h\_t)$$

然而，必須強調的是，此目標函數僅針對「觀測分佈（Observational distribution）」進行擬合，尚未涉及干預（Intervention）與因果化（Causalization）。

#### **3.1.7 Theoretical Value and Causal Transition**

本模組的核心理論貢獻在於：成功建立了一個不包含混淆變數捷徑（Confounder-free shortcut）的使用者狀態表示。透過在前期嚴格隔離 price、salesRank 等經濟與曝光變數，並堅守純 ID 驅動的表徵學習，我們為後續模組提供了一個乾淨且純粹的基底。這使得模組 3.2 得以在此基礎上，透過結構因果模型（SCM）引入經濟學變數，進行真正的去混淆與反事實推論（Counterfactual reasoning）。

---

**修改重點說明**：

1. **3.1.2** 結尾補上了「完全摒棄外部語意的介入」，強化特徵隔離的純粹性。  
2. **3.1.3 的 Item Embedding** 做了最關鍵的改寫，明確宣告這組 ID Embedding 就是貫穿全局的唯一商品表徵，不會有後續的語意擴充。  
3. **3.1.7** 稍微修改了敘述，將原本的「模組 B」統一名稱為「模組 3.2」，並再次點題「堅守純 ID 驅動的表徵學習」。

### **3.2 Causal Deconfounding Module**

本節將介紹 CEINN 框架的核心：因果去混淆模組（Causal Deconfounding Module）。有別於傳統模型致力於擬合觀測數據，本模組的統計目標是修正序列推薦中固有的選擇偏差，將模型從被動的觀測預測（Observational prediction）推向具備介入意識（Intervention-aware）的反事實推論。

#### **3.2.1 Causal Inference View in Sequential Recommendation**

在傳統的序列推薦系統中，模型所優化的目標是條件機率 $P(Y \\mid X)$。然而，從因果推論的視角來看，這種觀測分佈無法代表真正的使用者偏好，因為使用者與物品的互動受到了系統曝光機制的嚴重干擾。換言之，觀測條件機率並不等於介入機率：

$$P(Y \\mid X) \\neq P(Y \\mid do(X))$$

為了解決此一統計落差，我們必須將序列推薦重新建構為一個因果推論問題，透過 $do$-calculus 或逆機率加權來估計真實的偏好因果效應。

#### **3.2.2 Problem Formulation: The Exposure Bias**

在真實世界的觀測資料中，我們所收集到的互動矩陣極度稀疏。當我們觀測到 $Y\_{ui} \= 0$ 時，傳統模型往往將其視為「使用者 $u$ 不喜歡物品 $i$」（Negative feedback）。然而，這混淆了「偏好」與「曝光」。實際上，觀測到的互動機率應被拆解為曝光機率與真實偏好機率的乘積：

$$P(Y\_{ui}=1) \= P(E\_{ui}=1) \\cdot P(Y\_{ui}=1 \\mid E\_{ui}=1)$$

其中 $E\_{ui} \\in \\{0, 1\\}$ 為潛在變數，表示物品 $i$ 是否對使用者 $u$ 曝光。冷門物品或價格過高的物品往往 $P(E\_{ui}=1)$ 極低，導致其 $Y\_{ui}=0$ 僅是缺乏曝光所致。

#### **3.2.3 Structural Causal Model for Sequential Recommendation**

為正式化上述假設，我們建立一個有向無環圖（DAG）形式的結構因果模型（SCM）。定義圖中包含三個核心節點：

$$Z \\rightarrow E \\rightarrow Y \\quad \\text{and} \\quad Z \\rightarrow Y$$

其中 $Z$ 代表混淆變數（Confounders，如流行度或市場偏差），$E$ 為系統曝光決策，$Y$ 為最終的使用者互動。此因果圖明確指出：混淆變數 $Z$ 不僅直接影響了物品是否被曝光（$Z \\rightarrow E$），也可能直接影響使用者的互動決策（$Z \\rightarrow Y$，例如從眾效應）。若不切斷 $Z$ 對 $E$ 的依賴（即 Backdoor path），模型將無可避免地學習到偽相關（Spurious correlation）。

#### **3.2.4 Dataset-Aware Confounder Construction**

由於混淆變數 $Z$ 在不同場景下的可觀測性不同，我們針對兩種主流資料集設計專屬的混淆變數建構策略：

* **Amazon Beauty**: 該資料集包含豐富的元資料（metadata）。我們利用 salesRank 作為流行度混淆變數的代理指標（Proxy）。考慮到排名的長尾分佈與雜訊特性，我們對其進行對數轉換與分位數分桶（Quantile bucketization）：  
  $$Z\_i \= \\text{Bucket}(\\log(\\text{salesRank}\_i))$$  
* **MovieLens 10M**: 該資料集缺乏外部排名資訊。為避免未來資訊洩漏（Future leakage），我們僅使用時間點 $t$ 之前的歷史互動軌跡，動態計算物品 $i$ 的滾動互動頻率作為代理變數：  
  $$Z\_i(t) \= \\text{interaction frequency of item } i \\text{ before time } t$$

#### **3.2.5 Propensity Score Estimation**

為了進行偏差修正，我們必須估計每個物品的傾向分數（Propensity Score）$p\_i$，即給定混淆變數 $Z\_i$ 時，物品獲得曝光的條件機率：

$$p\_i \= P(E\_i=1 \\mid Z\_i)$$

在架構設計上，我們刻意採用淺層多層感知機（Shallow MLP）或 Logistic Regression 作為傾向估計器（Propensity Estimator）。其原因在於，若使用過度深層的神經網路，模型極易對流行度特徵產生過擬合（Overfitting）；而傾向分數估計本質上是一種密度比估計（Density ratio estimation），保持模型的適度容量（Capacity）有助於獲得更穩健的估計值。

#### **3.2.6 Inverse Propensity Scoring (IPS) Objective**

取得傾向分數後，我們將模組 3.1 的觀測損失函數（Observational loss）升級為逆傾向分數加權（Inverse Propensity Scoring, IPS）損失函數：

$$\\mathcal{L}\_{IPS} \= \- \\sum\_{u, i} \\frac{y\_{ui}}{p\_i} \\log \\hat{y}\_{ui}$$

在統計意義上，這是一種逆向抽樣修正（Inverse sampling correction）。對於流行度極低的冷門物品，其 $p\_i$ 較小，因此在損失函數中會獲得較大的權重補償；反之亦然。這強迫模型將注意力從「熱門但無關」的捷徑，轉移至「冷門但真實相關」的特徵上。

#### **3.2.7 Variance Reduction: Clipped and Self-Normalized IPS**

純粹的 IPS 方法在實務上存在變異數過高（High variance）的問題。特別是當冷門物品的傾向分數極端趨近於零（$p\_i \\to 0$）時，其倒數 $1/p\_i$ 會引發梯度爆炸。為穩定訓練過程，我們引入兩種變異數縮減技術。首先是截斷型 IPS（Clipped IPS），設定上限閾值 $\\tau$：

$$w\_i^{(clipped)} \= \\min\\left( \\frac{1}{p\_i}, \\tau \\right)$$

其次是自平滑 IPS（Self-normalized IPS），透過全局權重歸一化來穩定期望值：

$$w\_i^{(SN)} \= \\frac{1/p\_i}{\\sum\_j 1/p\_j}$$

實際訓練時，可依據資料的長尾嚴重程度切換這兩種權重策略。

#### **3.2.8 Adversarial Deconfounding on Latent Representations**

儘管 IPS 在樣本層面修正了偏差，但模組 3.1 所輸出的潛在狀態 $h\_t$ 仍可能在參數空間中隱性編碼了流行度資訊。特別是當模型嚴格遵循「特徵隔離原則」，**完全僅依賴純 ID 序列萃取偏好狀態時**，ID Embedding 極易淪為歷史流行度記憶的載體。

為了達成徹底的表示層級去混淆（Representation Deconfounding），我們引入對抗性正則化（Adversarial Regularization）。我們構建一個鑑別器 $D$，試圖從潛在狀態 $h\_t$ 預測其對應的混淆變數類別 $Z$：

$$\\mathcal{L}\_{adv} \= \- \\text{CE}(D(h\_t), Z)$$

同時，序列編碼器（Backbone）的目標是最大化此損失，透過梯度反轉層（Gradient Reversal Layer）使得鑑別器無法區分狀態的流行度屬性。**由於去混淆優化完全集中在「純 ID」表徵上，排除了外部語意的干擾，反而讓因果推論的實驗變因更加純粹。** 此對抗過程強制執行了條件獨立性 $h\_t \\perp Z$，確保 User-state 成為真正純粹的偏好表示。

#### **3.2.9 The Unified Causal Objective**

綜合上述所有因果推論與去混淆機制，模組 3.2 最終的統一優化目標函數定義如下：

$$\\mathcal{L} \= \\mathcal{L}\_{IPS} \+ \\lambda\_{adv}\\mathcal{L}\_{adv} \+ \\lambda\_{reg}\\|\\theta\\|^2$$

其中 $\\lambda\_{adv}$ 控制對抗去混淆的強度，$\\lambda\_{reg}$ 為標準的 L2 正則化權重。透過此統一目標，CEINN 成功將序列推薦從單純的歷史模式比對，昇華為具備反事實推理能力的動態偏好估計系統。

---

**修改重點說明：**

1. **模組名稱連貫性**：將原先文稿中的「模組 A」與「模組 B」統一修正為「模組 3.1」與「模組 3.2」，以符合論文的章節編號。  
2. **強化 3.2.8 的論述**：依照您的建議，在 3.2.8 段落中加入粗體字部分，明確點出「純 ID 序列萃取」的特性，這不僅解釋了為何需要對抗性正則化（因為 ID 容易記憶流行度），同時也將移除 3.4 模組（語意特徵）的劣勢轉化為實驗設計上「控制變因純粹」的學術優勢。

### **3.3 Economics-Informed Utility & Budget Module**

本節將探討 CEINN 框架中的決策核心：經濟學啟發的效用與預算模組（Economics-Informed Utility & Budget Module）。我們將捨棄傳統序列推薦中單純計算「使用者與物品相似度」的範式，轉而引入離散選擇模型（Discrete Choice Model），將推薦視為一場在特定情境下的效用最大化（Utility Maximization）過程。

#### **3.3.1 Statistical Learning Positioning: From Preference to Choice Theory**

在傳統的機器學習設定中，模型傾向於直接擬合條件機率 $P(y=1 \\mid u, i)$，將高互動頻率等同於高偏好。然而，在真實世界的微觀經濟學視視角下，使用者 $u$ 在時間 $t$ 選擇物品 $i$，是因為該物品在所有候選集合中提供了最大的淨效用。我們將此決策過程建模為一個隨機效用模型（Random Utility Model, RUM）的神經網路化版本：

$$i^\* \= \\arg\\max\_{i \\in \\mathcal{I}} U(u, i, t)$$  
其中 $U(u, i, t)$ 為使用者對候選物品的主觀總效用。此模組的統計機器學習定位，旨在還原決策過程中的權衡機制，而非僅作簡單的二元分類或迴歸。

#### **3.3.2 The Necessity of Utility-Cost Disentanglement**

若模型不具備效用與成本的解耦合（Disentanglement）能力，極易產生兩種致命偏差：「價值偏差（Value Bias）」，即誤認所有高互動頻率的商品皆具有高內在價值；以及「成本盲視（Cost Blindness）」，即忽略了低價格或低選擇摩擦力對促成互動的巨大貢獻。為此，我們將總效用嚴格拆解為價值（Value）與成本（Cost）的差值：

$$U(u, i, t) \= V(u, i, t) \- \\lambda\_u C(i, t)$$  
其中 $V(u, i, t)$ 代表使用者對物品的內在匹配價值，$C(i, t)$ 為獲取該物品的成本或選擇阻力，而 $\\lambda\_u$ 則為該使用者對成本的個人敏感度（Cost sensitivity）。

#### **3.3.3 Explicit Economic Cost Modeling (Amazon Beauty Scenario)**

對於具備豐富商業元資料（Metadata）的電子商務場景（如 Amazon Beauty 資料集），我們能直接觀測到明確的金錢與市場成本。基於 price、categories、brand 以及 salesRank 等實證變數，我們建構顯性經濟成本函數：

$$C(i, t) \= \\alpha\_1 \\log(\\text{price}\_i) \+ \\alpha\_2 \\phi(\\text{category}\_i) \+ \\alpha\_3 \\psi(\\text{brand}\_i) \+ \\alpha\_4 \\eta(\\text{salesRank}\_i) \+ \\alpha\_5 \\big(\\text{price}\_i \\times \\phi(\\text{category}\_i)\\big)$$  
此公式不僅捕捉了價格的絕對阻力（$\\alpha\_1$），更引入了交互作用項（$\\alpha\_5$）。其經濟學意義在於：同樣的價格標籤在不同品類（如高階精華液與一般洗面乳）中所代表的預算壓力截然不同。**為了在有限時間內實現高效訓練並避免複雜的文本語意運算，此處的品類表徵 $\\phi(\\text{category}\_i)$ 與品牌表徵 $\\psi(\\text{brand}\_i)$ 嚴格摒棄多模態文本模型，直接透過簡單的 One-hot 嵌入查找表（Embedding Lookup Table）取得**，在保持經濟學模型嚴謹度的同時，大幅降低工程實作複雜度。

#### **3.3.4 Implicit Behavioral Friction Modeling (MovieLens Scenario)**

針對如 MovieLens 10M 般缺乏真實價格資訊的純內容消費場景，我們將經濟學中的「成本」概念泛化為行為經濟學中的「選擇摩擦力（Choice Friction）」與「邊際效用遞減（Diminishing Marginal Utility）」。考量到社群標籤（Tags）極度稀疏且充满雜訊，**本研究聚焦於高質量的電影類型（Genres）與時間序列，將隱性行為成本函數精簡定義如下**：

$$C(i, t) \= \\beta\_1 \\cdot \\text{GenreRed}(i, \\mathcal{H}\_{u, \<t}) \+ \\beta\_2 \\cdot \\text{RecencyPress}(t) \+ \\beta\_3 \\cdot \\text{PopPress}(i)$$  
其中，$\\text{GenreRed}(i, \\mathcal{H}\_{u, \<t})$ 計算候選電影與使用者歷史觀影集合 $\\mathcal{H}\_{u, \<t}$ 在類型上的重疊冗餘度。**實務上，該項直接透過計算候選電影類型的二進位向量（Binary Vector）與使用者歷史消費類型頻次向量的傑卡德相似度（Jaccard Similarity）或簡單內積取得。** 若使用者短時間內連續消耗特定類別，疲勞效應（Satiation）將導致該類別新物品的行為成本急遽上升，從而促使模型推薦具備探索性（Exploration）的內容。

#### **3.3.5 Subjective Value and User-Specific Cost Sensitivity**

在確立了成本項後，我們定義內在價值 $V(u, i, t)$。**為了防止特徵糾纏，物品的表徵在此階段嚴格排除價格、品類等成本變數，直接採用模組 3.1 中所定義、純粹由互動歷史學得的商品唯一 ID 嵌入向量 $E\_i(i)$。** 我們將其與模組 3.2 輸出之具備因果去混淆特性的潛在狀態 $h\_t$ 進行匹配與非線性映射：

$$V(u, i, t) \= f\_\\theta(h\_t, E\_i(i))$$  
此外，不同使用者對成本的承受度存在異質性。我們將成本敏感度 $\\lambda\_u$ 建構為使用者潛在輪廓（Latent profile）的函數：

$$\\lambda\_u \= \\sigma(w^\\top z\_u)$$  
其中 $z\_u$ 為由使用者歷史互動聚合而成的個性化向量，$\\sigma$ 為 Sigmoid 函數以確保非負性。此設計使得 $\\lambda\_u$ 成為一個高度可解釋的潛在參數，動態調節價格或疲勞感對決策的影響權重。

#### **3.3.6 Optimization Objective: Neuralized Discrete Choice Model**

基於上述的效用拆解，我們假設決策過程中的未觀測擾動服從獨立同分配的 Gumbel 分佈（Type I Extreme Value Distribution），從而將最大效用選擇成功推導為標準的 Softmax 機率形式。給定時間 $t$ 的去混淆使用者狀態 $h\_t$，選擇物品 $i\_t$ 的機率為：

$$P(i\_t \\mid h\_t) \= \\frac{\\exp(U(u, i\_t, t))}{\\sum\_{j \\in \\mathcal{I}} \\exp(U(u, j, t))}$$  
我們透過最小化負對數似然損失（Negative Log-Likelihood Loss）來優化此選擇模型：

$$\\mathcal{L}\_C \= \-\\log P(i\_t \\mid h\_t)$$  
至此，模組 3.3 成功將傳統的偏好學習轉化為具備微觀經濟學基礎的效用決策過程，使得模型能在「使用者真實偏好」與「現實阻力」之間取得最佳的推薦平衡。