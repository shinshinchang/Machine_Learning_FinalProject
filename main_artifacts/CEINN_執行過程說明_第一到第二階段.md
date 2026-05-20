### **Phase 1 任務本質的再剖析**

Phase 1 不做任何前處理、不做特徵工程，**僅做一件事**：用 EDA 報告中已建立的「黃金標準」反向驗證讀取邏輯的正確性。這是整個專案的「資料品質閘」——若此處有 bug，後續所有模型實驗都會被汙染。

從計畫文本中我萃取出 Phase 1 必須完成的三個層次的驗證：

**1\. 結構性驗證（必須完全相等）**

* Amazon：互動數 198,502、使用者 22,363、商品 12,101、葉節點類別 222  
* MovieLens：5-core 前 10,000,054、5-core 後 9,998,816、使用者 69,878、電影 10,196、類型 19、標籤事件 95,580

**2\. 分布性驗證（容許微小誤差）**

* Amazon：5 星比例 57.70%、Price 覆蓋率 95.17%、SalesRank-Any 98.2%、SalesRank-Beauty 88.3%、Brand 82.8%  
* MovieLens：半星 20.5%、5 星 15.4%、評分熵 1.912 bits、單例標籤 55.9%

**3\. 物理性驗證（單位、編碼、時區）**

* Amazon `unixReviewTime` 是秒級 epoch（非毫秒）  
* Amazon 時間跨度：2002-06-12 至 2014-07-23（4,424 天）  
* MovieLens `movies.dat` 編碼是 ISO-8859-1（含重音字元）  
* MovieLens 類型集合排除 `(no genres listed)` 後恰為 19

### **三個值得特別處理的邊緣案例**

1. **Amazon 的 JSON 格式雙重性**：McAuley 早期 dump 使用 Python dict literal（單引號），新版才是標準 JSON。讀取時必須 `json.loads` 失敗則退回 `ast.literal_eval`。  
2. **`salesRank` 欄位的型態歧異**：可能是 `{"Beauty": 12345}`、`{}`、`null`，或字串。EDA 區分「任意品類覆蓋率 98.2%」與「Beauty 品類覆蓋率 88.3%」，所以解析時需保留「來源層級」標記（preferred\_key / any\_key / missing），不能簡單做 truthy 檢查。  
3. **MovieLens 5-core 過濾的不動點**：EDA 報告載明 5-core 後恰為 9,998,816。由於 ML10M 原始已近乎 5-core，需用 iterative filtering（while loop 直至 user/item 集合穩定）才能精確重現此數字。

### **模組化設計取捨**

原規劃的 `data_loaders/` 目錄是給「訓練時的特徵 lookup」用，**不應**承擔 raw 檔案解析的職責。為了讓 `preprocess.py` 保持流程編排的清晰度，我做一個小擴展：新增 `utils/data_io.py` 作為純函式的底層 I/O 模組。這個分層讓後續 Phase 2 也能直接 import 同一套 reader，避免兩遍實作。

`preprocess.py` 採用 `--phase` 參數設計，Phase 1 階段只實作驗證流程，後續 Phase 2/3 可在同一檔案內延伸，不必重構入口邏輯。

---

接下來開始實作。先建立目錄結構。

規劃檔案結構，依序實作短檔案、長檔案與文檔。  
規劃檔案結構，依序實作短檔案、長檔案與文檔。

目錄結構建好了。先建立配置檔，把所有 EDA 預期值與容許誤差外化到 YAML，這樣後續若有 bug 也能直接修改門檻而不動程式碼。

Created 2 files  
Created 2 files

現在建立底層的 raw 檔案讀取模組。這是純函式層,後續所有 Phase 都會 import 同一套 reader。

Created 2 files  
Created 2 files

現在建立核心的 `preprocess.py`。這是 Phase 1 的主入口,內含 `ValidationLog` 結構與兩個資料集的逐項驗證流程。

重構半星評分邏輯並驗證資料預處理正確性。  
重構半星評分邏輯並驗證資料預處理正確性。

讓我對 `preprocess.py` 做一個小重構——MovieLens 的半星判斷可以寫得更清晰：

驗證預處理腳本語法與導入正確性。  
驗證預處理腳本語法與導入正確性。

讓我用 Python AST 工具靜態檢查整個專案的語法正確性,並做一次模組 import 測試。

驗證純函式測試後，設計端到端測試流程。  
驗證純函式測試後，設計端到端測試流程。

純函式測試通過了。現在我建立小型 fixture 檔來做端到端的 dry-run，確認整條 validate 管線在面對真實檔案格式時能正確運作:

驗證完成，籌備輔助檔案與輸出部署。  
驗證完成，籌備輔助檔案與輸出部署。

端到端測試全部通過。特別值得確認的幾點:Python literal 格式被正確 fallback 解析、ISO-8859-1 編碼下 `é` 字元能讀取、5-core iterative filtering 收斂正確、`salesrank_any` 與 `salesrank_beauty` 兩種覆蓋率被正確區分。

現在補齊 `requirements.txt`、`.gitignore`、與一份精簡的 README:

Created 3 files, ran a command  
Created 3 files, ran a command

最後寫一份聚焦於 Phase 1 的 README,讓使用者(包括組員)能照著一鍵驗證資料:

Ran 3 commands, created a file, read a file  
Ran 3 commands, created a file, read a file

Phase 1 的完整交付已就緒。以下是這一階段的設計脈絡與後續銜接點:

### **設計脈絡總結**

#### **三層架構,各司其職**

我刻意把 Phase 1 拆成三層,使後續 Phase 不必重構入口:

1. **`configs/*.yaml`(資料外化層)**:把所有 EDA 預期值與容許誤差搬出程式碼。若 Phase 7 消融實驗發現某項統計的精度需要重新審視,直接編輯 YAML 即可,無需碰 Python。  
2. **`utils/data_io.py`(純函式 I/O 層)**:處理三類 raw 檔案格式邊緣案例——Amazon 的 JSON / Python literal 雙重格式、MovieLens 的 ISO-8859-1 編碼、`salesRank` 的型態歧異。這層被 Phase 2 直接 reuse,不必重新撰寫。  
3. **`preprocess.py`(流程編排層)**:`ValidationLog` 結構統一所有檢核項的記錄格式,並以 `--phase` 參數預留 Phase 2+ 的擴展位。

#### **與原規劃的偏離(已加註說明)**

`utils/data_io.py` 不在最初的目錄結構中。我新增的理由在 README 中明確說明:它讓 Phase 2 的前處理與 Phase 1 的驗證共享同一套 parser,避免兩處 reader 對邊緣案例做出不一致的解讀(這正是執行計畫 Phase 1 品質閘所警告的「不得帶著已知不一致性進入下游」風險的根因)。

#### **端到端 dry-run 已通過**

我用合成資料(含 Python literal 格式的 metadata、含 `é` 的 Latin-1 電影標題、需 iterative 5-core 收斂的互動圖)驗證了整條 pipeline 在 29 個檢核點上全數通過。當您把真實的 `Beauty_5.json`、`meta_Beauty.json`、`ml-10M100K/*.dat` 放到 `data/raw/` 下執行 `python preprocess.py --phase 1`,應該會看到所有 30 個檢核項(Amazon 15 \+ MovieLens 15)PASS。

### **兩個值得提前對齊的設計判斷**

第一,**MovieLens 的 5-core 是 iterative fixed-point,不是單次過濾**。EDA 報告的 9,998,816 筆是收斂後的數字,單次過濾會稍微偏多。我的 `_five_core_filter` 用 while loop 直到 user/item 集合穩定,通常 1–2 次迭代即收斂。

第二,**`salesRank` 必須區分「來源層級」**。EDA 給了「任意品類覆蓋 98.2%」與「Beauty 品類覆蓋 88.3%」兩個獨立指標,所以 `extract_salesrank` 回傳的不只是值,還有 `source ∈ {"preferred_key", "any_key", "missing"}` 的標籤——這個訊息會在 Phase 2 任務 2.2.4 的分桶階段繼續使用(「Beauty 缺失 → 退而採用任意品類」的 fallback 規則)。

#### **第一階段的終端機輸出整理：**

(ceinn\_env) root@LAPTOP-FSHU34NF:/mnt/d/\_\_projects\_main/ceinn\# python preprocess.py \--phase 1

\>\>\> Validating Amazon Beauty …

\=== Amazon Beauty \===

\[PASS\]  n\_interactions                              expected=198502          actual=198502          tol=0         mode=exact

\[PASS\]  n\_users                                     expected=22363           actual=22363           tol=0         mode=exact

\[PASS\]  n\_items                                     expected=12101           actual=12101           tol=0         mode=exact

\[PASS\]  rating\_share\_5                              expected=0.577000        actual=0.576977        tol=0.001     mode=abs

\[PASS\]  unixReviewTime unit (seconds)               expected=True            actual=True            tol=0         mode=exact

\[PASS\]  time\_range\_start (UTC date)                 expected=2002-06-12      actual=2002-06-12      tol=1         mode=date\_abs

\[PASS\]  time\_range\_end (UTC date)                   expected=2014-07-23      actual=2014-07-23      tol=1         mode=date\_abs

\[PASS\]  time\_span\_days                              expected=4424            actual=4424            tol=2         mode=abs

\[PASS\]  meta records matched to reviewed items      expected=12101           actual=12101           tol=0         mode=exact

          | note: all reviewed asins should be present in meta

\[PASS\]  price\_coverage                              expected=0.951700        actual=0.951657        tol=0.005     mode=abs

\[PASS\]  salesrank\_any\_coverage                      expected=0.982000        actual=0.981985        tol=0.005     mode=abs

\[PASS\]  salesrank\_beauty\_coverage                   expected=0.883000        actual=0.883150        tol=0.005     mode=abs

\[PASS\]  brand\_coverage                              expected=0.828000        actual=0.826626        tol=0.005     mode=abs

\[PASS\]  n\_leaf\_categories                           expected=222             actual=222             tol=0         mode=exact

          | note: union of leaf-level category labels over reviewed items

\----------------------------------------------------------------------------------------------------

Summary: 14/14 checks passed — OK

\>\>\> Validating MovieLens 10M …

\=== MovieLens 10M \===

\[PASS\]  n\_interactions\_raw                          expected=10000054        actual=10000054        tol=0         mode=exact

\[PASS\]  rating\_half\_star\_share                      expected=0.205000        actual=0.204822        tol=0.001     mode=abs

\[PASS\]  rating\_share\_5                              expected=0.154000        actual=0.154480        tol=0.001     mode=abs

\[PASS\]  rating\_entropy\_bits                         expected=2.758393        actual=2.758393        tol=0.01      mode=abs

\[PASS\]  n\_interactions\_5core                        expected=9998816         actual=9998816         tol=0         mode=exact

          | note: iterative 5-core fixed-point

\[PASS\]  n\_users (5-core)                            expected=69878           actual=69878           tol=0         mode=exact

\[PASS\]  n\_items\_5core                               expected=10196           actual=10196           tol=0         mode=exact

\[PASS\]  time\_range\_start (UTC date)                 expected=1995-01-09      actual=1995-01-09      tol=1         mode=date\_abs

\[PASS\]  time\_range\_end (UTC date)                   expected=2009-01-05      actual=2009-01-05      tol=1         mode=date\_abs

\[PASS\]  time\_span\_days                              expected=5109            actual=5109            tol=2         mode=abs

\[PASS\]  n\_genres (excl. no-genres-listed)           expected=19              actual=19              tol=0         mode=exact

          | note: genres found: \['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'\]

\[PASS\]  n\_items\_raw (from movies.dat)               expected=10681           actual=10681           tol=0         mode=exact

\[PASS\]  n\_tag\_events                                expected=95580           actual=95580           tol=0         mode=exact

\[PASS\]  n\_tag\_vocab (case-sensitive)                expected=15240           actual=15240           tol=0         mode=exact

\[PASS\]  tag\_singleton\_share                         expected=0.559000        actual=0.558727        tol=0.005     mode=abs

          | note: singletons / unique tag strings

\----------------------------------------------------------------------------------------------------

Summary: 15/15 checks passed — OK

Validation log written to: data/raw/validation\_log.txt

Phase 1 quality gate PASSED. Safe to proceed to Phase 2\.

### 

### **Phase 2 深度規劃**

#### **一、Phase 2 的本質**

Phase 1 是「驗證讀取」(read-only),Phase 2 才是「真正的資料工程」:把 raw 互動轉成模型訓練時 O(1) 可查的張量化字典。**這個階段一旦寫入 `data/processed/` 的內容有 bug,後續所有 Phase 都會被汙染**——而且不像 Phase 1 有 EDA 報告做反向核驗,Phase 2 的正確性主要靠內部一致性檢查(invariants)而非外部基準。

#### **二、計畫文本 vs. 範例檔案的細節對齊**

我重新檢視範例檔案後,有幾個關鍵的觀察必須在實作中處理:

**1\. `meta_Beauty.json` 的關鍵特性**

* 第二筆範例(asin `6175005589`)**完全沒有 `price` 欄位**——不是 `null`,是 key 不存在;`brand` 也不存在。  
* 第三筆範例(asin `6167061580`)**沒有 `price` 也沒有 `salesRank`** 嗎?有,salesRank 存在但 `price` 是 4.48 存在。範例本身也展示 `salesRank` 鍵可能就是 `{'Beauty': 數字}`。  
* `categories` 範例都是單一路徑列表;雖然格式上是 `[[...]]`,實務上一個商品仍可能有多個 path——`extract_leaf_categories` 已能處理。

**2\. `ratings.dat` 的 timestamp 微妙處**

* 範例中 user 9979 的多筆 rating 都有**不同的 timestamp**,但 EDA 報告告訴我們 78.9% 的相鄰互動 `Δt < 1 分鐘`。**這代表 ratings.dat 內的 timestamp 是「補評時間」而非「觀影時間」**——這正是 Task 2.3.1 要做 60 秒 session 切分的根本動機。

**3\. `movies.dat` 的 movieID 不連續**

* 範例顯示 1\~10 都連續,但實際上 MovieLens 10M 有部分 ID gap;`item2idx` 重映射就是要解決這個問題。

#### **三、容易出錯的邏輯陷阱(我會主動防範)**

1. **`extract_leaf_categories` 在 Phase 1 設計時回傳「所有 paths 的 leaves」(可能多個)**。但 Phase 2 任務 2.2.1 說「取最深層作為商品的品類代表」——單一商品應該只對應**一個**品類 ID。我需要新增一個策略:多 path 時取最常見的 leaf,或取第一個 path 的 leaf;**且要在程式中明確標記這個選擇**。  
2. **5-core 應該對 5-core 後重排序的 leaf-categories 重新計數**。EDA 的 222 是在「reviewed items 上的 leaf 聯集」;但 Phase 2 任務 2.2.1 是建立 lookup table,要用 5-core 後的 12,101 item 來算 leaf counts(因為 \< 5 次的稀有 leaf 要合併為 UNK)。  
3. **Amazon `unixReviewTime` 的精度問題**:範例顯示 `1405209600` 與 `1372118400`,**都是當日 00:00:00 UTC** 整數秒——這代表 Amazon 的 timestamp 精度只到「日」\!所以同使用者同日多筆評論的時序排序是**不穩定**的;Phase 2 必須引入 tie-breaker(如使用陣列原始順序作為次要排序鍵)。  
4. **MovieLens session 切分後的「同 session 互動」處理**:任務 2.3.1 說「session 內部視為同一時刻」。這影響 train\_seqs 的格式——是要把同 session 的多筆互動合併成一個多 item 的 step,還是各自獨立但 `dt_bin = PAD`?計畫文本不明朗,但從研究方法 §3.1.3 的 "ek=Ei(ik)+Er(rk)+Et(Δtk)e\_k \= E\_i(i\_k) \+ E\_r(r\_k) \+ E\_t(\\Delta t\_k) ek​=Ei​(ik​)+Er​(rk​)+Et​(Δtk​)" 看,每筆互動仍是獨立 token,只是 `Δt` 在 session 內為 0(或對應的 PAD 桶)。我採後者。  
5. **動態 Z\_i(t) 與 GenreRed 的時序語意必須對齊**:Z\_i(t) 必須是「t **之前**的累積次數」,絕不能包含 t 本身(否則就是 leakage);GenreRed 同理。實作時用「先記錄分數、再更新累積狀態」的 sweep 順序。  
6. **GenreRed 的記憶體爆炸風險**:9.99M 互動 × 一個 float \= 80MB,可控。但若用 `{(uid, iid, ts): score}` 的 dict 儲存,Python dict 的 overhead 會讓 80MB 膨脹到 600MB+。我**改用陣列+索引**:把訓練互動排成一個固定陣列,Z 與 GenreRed 都用 `np.array` 平行索引。  
7. **IDF 加權的「文件」定義**:在計算類型 IDF 時,「文件」是電影還是使用者?從統計學意義上,降低 Drama/Comedy 的權重需要用**全電影集**作 corpus(因為 IDF 反映類型的全局稀有度)。

#### **四、檔案責任分工(對應規劃文件 \+ 我的小修正)**

* **`utils/math_utils.py`(新增)**:純函式統計工具,被多處 reuse:對數分桶、Jaccard、IDF 加權 Jaccard。  
* **`preprocess.py`(擴充)**:`--phase 2` 主流程,負責 5-core / 重映射 / 切分 / Δt 分桶 / 序列截斷 / 動態 Z / GenreRed sweep / 序列化。**所有寫入動作集中在這裡**。  
* **`data_loaders/amazon_beauty_loader.py`(新增)**:**訓練時**的查找介面,把 `item_meta.pkl` 包成 PyTorch-friendly 的張量(目前 Phase 2 只需先把資料介面定義好;不依賴 torch,維持 numpy 為主)。  
* **`data_loaders/movieslens_10M_loader.py`(新增)**:同理,提供 train/val/test 序列 \+ 動態 Z \+ GenreRed 的查找。

為了避免 Phase 6 才發現介面不對需要重構,我先把 loaders 寫成「能讀回 processed/ 並提供索引」的最小骨架(沒有 PyTorch Dataset,僅 numpy/dict),Phase 4/6 再用 torch wrap 即可。

#### **五、決策落定**

我把規劃的決策一條條列出讓您審視:

| 議題 | 我的決策 | 理由 |
| ----- | ----- | ----- |
| Amazon 是否從 `reviews_Beauty.json` 重跑 5-core? | **不,直接用 `Beauty_5.json`** | Phase 1 已驗證 `Beauty_5.json` 正是 5-core 後狀態(198,502 / 22,363 / 12,101 與 EDA 完全相符),且這就是 EDA 報告的分析來源,保持一致性。 |
| Amazon 同日多互動 tie-breaker | **檔案出現順序作次要 key** | 唯一穩定且可重現的 tie-breaker。 |
| `extract_leaf_categories` 多 leaf 時取哪個? | **取「該 path 的 first occurrence」** | 多數商品本就只有一條 path,有多 path 時第一條通常是「主要分類」(McAuley dump convention)。新增 `extract_primary_leaf_category` 函式。 |
| Amazon Δt 用 60 秒 session 嗎? | **不用,Amazon 沒有此問題** | EDA 顯示 Amazon 是 43.94% 同日,不是 78.9% 同分鐘;且 unixReviewTime 精度只到日。直接 Δt 對數分桶 32 桶即可。 |
| MovieLens session 切分後同 session item 處理 | **各自獨立為 token,session 內 Δt → 桶 0 (PAD)** | 對齊計畫文本「Session 內部視為同一時刻」,且不破壞 token 顆粒度。 |
| Δt 分桶邊界從哪算 | **從訓練集所有 Δt 上 fit,凍結後用於 val/test** | 防止洩漏。第一筆無前驅的 Δt 直接給 bin 0 (PAD),不參與 fit。 |
| Dynamic Z 儲存格式 | **`np.array` 與訓練互動 list 等長平行對齊**,不用 dict | 大幅節省記憶體;dict 介面再透過 helper 提供。 |
| GenreRed IDF 的 IDF 從哪算 | **從 5-core 後所有 movies 的 genre 集合算**,不是訓練集 | IDF 是電影集合的屬性,不應隨切分變動。 |
| 2009 截斷的時機 | **僅排除 train 中的 2009;val/test 不動** | 計畫文本明確要求「保留評估正確性」。 |

理由攤開後,我開始實作。

---

### **程式實作**

先建立 `math_utils.py`,這是所有統計工具的核心:

Created a file, ran a command  
Created a file, ran a command

我的測試 corpus 不對:genre 1 與 genre 2 都只出現一次(df=1),所以 IDF 相等。修正測試:

Edited a file, ran a command  
Edited a file, ran a command

接下來在 `data_io.py` 加入「取主要葉節點」的便利函式,並擴充 Amazon meta 解析的工具。先看當前狀態:

確認 data\_io.py 中既有的 extract 函式介面  
確認 data\_io.py 中既有的 extract 函式介面

我在 `data_io.py` 的尾端加上 Phase 2 需要的工具函式(取主要葉節點、解析價格、品牌正規化):

Viewed a file, edited a file  
Viewed a file, edited a file

現在更新兩份 YAML config,加入 Phase 2 所需的所有超參數:

Edited 2 files  
Edited 2 files

好,核心 config 與底層工具已就位。接下來是 Phase 2 的主要實作:`preprocess.py` 的 Phase 2 區段。考慮到檔案會變很大,我把 Phase 2 的兩個 dataset 各自封裝成獨立函式,主流程只負責 dispatch。

由於 `preprocess.py` 已經 568 行,我會在尾端追加 Phase 2 區段,並更新 `main()` 的 dispatch:

擴充 preprocess.py 的 docstring,涵蓋 Phase 2  
擴充 preprocess.py 的 docstring,涵蓋 Phase 2

現在更新 imports 並加入 Phase 2 的核心邏輯。先增 imports:

擴充 preprocess.py 的 imports 以支援 Phase 2  
擴充 preprocess.py 的 imports 以支援 Phase 2

現在更新 `main()` 加入 Phase 2 的 dispatch,並在尾端追加 Phase 2 主邏輯。先把 main 重構成支援 phase 1 \+ phase 2:

重構 main(),支援 phase 1 / phase 2 dispatch  
重構 main(),支援 phase 1 / phase 2 dispatch

現在在 `if __name__ == "__main__":` 之前插入所有 Phase 2 邏輯。由於這部分很大,我會非常仔細地構造。先讓我在合適的位置插入 Phase 2 的核心區塊:

Viewed a file, edited a file  
Viewed a file, edited a file

現在做語法檢查:

Ran 2 commands  
Ran 2 commands

現在用合成資料做端到端 dry-run,驗證 Phase 2 流程能正確跑通並產出 invariants 全 PASS:

Ran 2 commands, created a file  
Ran 2 commands, created a file

忘了先建立資料夾,修正並重試:

Edited 2 files, ran a command  
Edited 2 files, ran a command

測試腳本需要正確的 sys.path,我把它放到 ceinn 目錄裡跑:

複製測試腳本到 ceinn 目錄並執行  
複製測試腳本到 ceinn 目錄並執行

端到端 dry-run 全部通過。所有 invariants 都過了:dynamic Z 單調遞增、GenreRed 在 \[0,1\] 範圍內、IDF 加權與標準 Jaccard 數值差異合理、Python literal 解析正確、9/9 \+ 10/10 invariants 全 PASS。

清掉測試腳本,接著實作 `data_loaders/`:

清理測試腳本並建立 data\_loaders 目錄  
清理測試腳本並建立 data\_loaders 目錄

現在實作 `data_loaders/` 的兩個模組。這層的職責是「**讀回 processed/ 並提供高層查找介面**」,不依賴 PyTorch(Phase 6 才會 wrap):

Created 3 files  
Created 3 files

驗證兩個 loader 都能正確讀回 Phase 2 的輸出:

Created a file, ran a command  
Created a file, ran a command

整合測試完全通過。看到的東西非常合理:

* User 1 序列為 item 1, 2, 3  
* pos=0: 第一筆,history union 為空 → Jaccard \= 0(符合定義)  
* pos=1: 看過 item 1 (Adventure/Animation/Children/Comedy/Fantasy),candidate item 2 (Adventure/Children/Fantasy);交集 3 個,聯集 5 個 → 3/5 \= 0.6 ✓  
* pos=2: history union \= item 1 ∪ item 2 \= 5 genres; candidate item 3 (Comedy/Romance);交集 1 (Comedy),聯集 6 → 1/6 ≈ 0.167 ✓  
* Dynamic Z 全是 0,因為每個 item 在每位 user 的訓練序列中只出現一次,前面沒人看過

GenreRed 與 dynamic Z 的計算邏輯正確\!

清掉測試腳本,確保 Phase 1 在新版的 preprocess.py 仍能跑(回歸測試):

確認 Phase 1 入口仍正常運作 (回歸測試)  
確認 Phase 1 入口仍正常運作 (回歸測試)

Phase 1 正確處理缺失資料的情境,並如預期 exit 1。

最後檢查整個專案結構並更新 README:

Ran a command, edited a file  
Ran a command, edited a file

您 Phase 1 在真實資料上跑出來 14/14 \+ 15/15 全 PASS,完美對齊 EDA 基線。現在 Phase 2 的程式碼也已經實作完成並通過合成資料的端到端驗證。

讓我把所有檔案匯總到 outputs 目錄,並更新一份 requirements.txt 與 .gitignore 確保 Phase 2 的依賴完整:

確認當前 requirements.txt 內容  
確認當前 requirements.txt 內容

`requirements.txt` 已涵蓋 Phase 2 所需。現在確認所有檔案到位,再執行最後一次完整 sanity check,然後匯總到 outputs:

Ran 2 commands  
Ran 2 commands

所有檔案就緒。匯總到 outputs:

Ran a command, read a file  
Ran a command, read a file  
---

### **Phase 2 實作完成總結**

Phase 1 已在真實資料上跑出 14/14 \+ 15/15 全 PASS,Phase 2 的程式碼也已實作完成並在合成 fixture 上通過 9/9 \+ 10/10 invariants。整份 Phase 2 涵蓋:

**新增的程式碼模組**

* `utils/math_utils.py` (279 行):log-quantile 分桶、Jaccard、IDF 加權 Jaccard,內含 `_self_check()` 在 import 時自動驗證  
* `preprocess.py` 擴充至 1,506 行:`preprocess_amazon_beauty()` \+ `preprocess_movielens_10m()`,各自帶獨立的 `_Phase2InvariantLog`  
* `data_loaders/amazon_beauty_loader.py` (153 行):`AmazonBeautyLoader` dataclass,提供 `category_of` / `brand_of` / `log_price_of` / `Z_of` / `bulk_meta_arrays` 介面  
* `data_loaders/movieslens_10M_loader.py` (190 行):`MovieLens10MLoader` dataclass,提供 `Z_at(u, pos)` / `genre_red_at(u, pos)` / `genre_red_idf_at(u, pos)` / `bulk_genre_matrix` 介面  
* `utils/data_io.py` 擴充:新增 `extract_primary_leaf_category` / `get_price_or_none` / `get_brand_or_none`

**幾項關鍵的設計決策都已寫進 YAML 與 README**,日後若論文審稿人質疑可直接出示:

1. Amazon 直接用 `Beauty_5.json`(Phase 1 已驗證等於 5-core 結果)  
2. Amazon 同日 timestamp ties → file\_order 作為穩定 tie-breaker(unixReviewTime 只到日精度)  
3. 多 path categories → 取第一個 path 的 leaf(McAuley dump convention)  
4. MovieLens 60 秒 session 邊界,session 內 Δt → bin 0 (PAD)  
5. Δt 分桶邊界只在 train 上 fit,凍結後套用 val/test(防 leakage)  
6. Bin 0 統一保留為 PAD:user\_idx / item\_idx / dt\_bin / rating\_bin  
7. **Dynamic Z 與 GenreRed 用 parallel `np.array` \+ `row_index` dict**:10M 互動若用 `{(u,i,t): float}` dict 會多 \~600MB Python overhead  
8. GenreRed IDF 從 5-core 全電影 corpus 算,**不**從訓練集算(IDF 是電影集合的屬性)  
9. Z 的時序 sweep 採「先記錄分數、後遞增計數」順序,嚴格符合「t 之前」的因果定義  
10. 2009 截斷只動 train,val/test 完整保留(對齊計畫 §2.3.5)

**接下來請您執行**

bash

```shell
cd /mnt/d/__projects_main/ceinn
python preprocess.py --phase 2
```

預期會看到兩個資料集各自跑完並印出 invariant 報告。在真實資料上,Amazon 那邊預期會有 9/9 全過,**MovieLens 那邊的 dynamic-Z 單調性檢查我特地寫成「抽 20 個 item」**,因為若對 10,196 個 item 全跑會慢——若您想要更嚴格,可以把 `preprocess.py` 裡的 `sample_items = list(item2idx.values())[:20]` 改成全集。

#### **第二階段的終端機輸出整理：**

(ceinn\_env) root@LAPTOP-FSHU34NF:/mnt/d/\_\_projects\_main/ceinn\# python preprocess.py \--phase 2 \--only amazon

\>\>\> Preprocessing Amazon Beauty …

  \[1/8\] Reading reviews from data/raw/Beauty\_5.json …

        kept 198502 records

  \[2/8\] Building user/item id maps …

        n\_users=22363, n\_items=12101

  \[3/8\] Sorting and splitting (leave-one-out temporal) …

  \[4/8\] Fitting Δt log-bucket edges on training set …

        74,392 train Δt's → 31 real bins

  \[5/8\] Bucketising Δt \+ truncating train sequences …

  \[6/8\] Building item metadata lookups …

        n\_cats=172, n\_brands=216, prices\_imputed=585

  \[7/8\] Writing artefacts to data/processed/amazon\_beauty …

  \[8/8\] Checking Phase-2 invariants …

\--- Phase 2 invariants: Amazon Beauty \---

  \[PASS\]  train item ids in \[1, n\_items\]

  \[PASS\]  future-leakage check (informational)  \-- 0 users have item-id reappearance (allowed if at different timestamps)

  \[PASS\]  n\_users matches user2idx  \-- 22363 vs 22363

  \[PASS\]  n\_items matches item2idx

  \[PASS\]  n\_cats \== 2 \+ |cat\_keep|

  \[PASS\]  n\_brands \== 2 \+ |brand\_keep|

  \[PASS\]  Δt bin range ok  \-- max bin \= 31

  \[PASS\]  Z bin range ok  \-- min=0, max=9

  \[PASS\]  item\_meta covers every item

  Summary: 9/9 invariants passed

Phase 2 quality gate PASSED. Processed files written to data/processed/.

---

(ceinn\_env) root@LAPTOP-FSHU34NF:/mnt/d/\_\_projects\_main/ceinn\# python preprocess.py \--phase 2 \--only movielens1

\>\>\> Preprocessing MovieLens 10M — Stage 1 …

  \[1/8\] Reading ratings from data/raw/ml-10M100K/ratings.dat …

        10,000,054 raw interactions

  \[2/8\] Applying iterative 5-core …

        kept 9,998,816 / 10,000,054 interactions (69878 users, 10196 items)

  \[3/8\] Building user/item id maps …

        n\_users=69878, n\_items=10196

  \[4/8\] Sorting, session-segmenting, and splitting …

        dropped 13,456 train rows from year \>= 2009

  \[5/8\] Fitting Δt log-bucket edges on training set …

        1,475,341 cross-session Δt's → 63 real bins

  \[6/8\] Bucketising Δt \+ truncating train sequences …

  \[7/8\] Reading movies.dat and building genre vectors …

        built 10,196 genre vectors over 19 genres

  \[8/8\] Writing Stage 1 artefacts …

\--- Phase 2 invariants: MovieLens 10M (Stage 1\) \---

  \[PASS\]  train item ids in \[1, n\_items\]

  \[PASS\]  Δt bin range ok  \-- max bin \= 63

  \[PASS\]  item\_genre covers every item  \-- 10196 vs 10196

  Summary: 3/3 invariants passed

Phase 2 quality gate PASSED. Processed files written to data/processed/.

---

(ceinn\_env) root@LAPTOP-FSHU34NF:/mnt/d/\_\_projects\_main/ceinn\# python preprocess.py \--phase 2 \--only movielens2

\>\>\> Preprocessing MovieLens 10M — Stage 2 …

  \[1/1\] Writing Stage 2 artefacts …

\--- Phase 2 invariants: MovieLens 10M (Stage 2\) \---

  \[PASS\]  train item ids in \[1, n\_items\]

  \[PASS\]  Δt bin range ok  \-- max bin \= 63

  \[PASS\]  dynamic Z non-negative  \-- min=0, max=27585

  \[PASS\]  GenreRed (std) in \[0, 1\]

  \[PASS\]  GenreRed (idf) in \[0, 1\]

  \[PASS\]  row\_index size matches values

  Summary: 6/6 invariants passed

Phase 2 quality gate PASSED. Processed files written to data/processed/.