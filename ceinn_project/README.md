# CEINN for MovieLens-1M (PyTorch)

這份程式是依照你上傳的 CEINN 研究計畫，做出一個**可執行、可訓練、可評估、可做消融實驗**的 PyTorch 版本。

## 目前支援
- MovieLens-1M 預處理
- leave-one-out 序列推薦切分
- CEINN 模型核心模組：
  - Transformer 序列編碼器
  - 語意 / 經濟雙流解耦
  - 動態參考點
  - 平滑版 Prospect Theory utility
  - Hyperbolic discount attention bias
  - SNIPS-style propensity weighting
  - latent confounder prototype backdoor adjustment
  - short-term / long-term heads
- HR@10 / NDCG@10 評估
- 消融實驗支援（Ablation Studies）：
  - `--ablation none` : 完整 CEINN 模型
  - `--ablation w/o_pt` : 無 Prospect Theory（直接用原始效用，無動態參考點）
  - `--ablation w/o_hd` : 無 Hyperbolic Discounting（基本因果注意力遮罩）
  - `--ablation w/o_mtl` : 無多任務學習（僅經濟頭，無語意頭）
  - `--ablation w/o_causal` : 無因果調整（無 SNIPS 權重、無潛在混淆因子）

## 重要說明
因為 MovieLens-1M 沒有「曝光紀錄」「加入收藏」「停留時間」「完整觀看率」這些欄位，所以研究計畫中的部分概念必須做工程近似：

1. **曝光 propensity**：用 item popularity 近似。
2. **長期價值標籤**：用較高評分（例如 >= 4）近似正向長期價值。
3. **短期誘惑標籤**：用一般互動事件（評分存在）近似。
4. **經濟屬性**：MovieLens 沒有價格，因此用 `genres / year / popularity` 當作 item side features。

也就是說，這份程式是「**以 MovieLens-1M 可以做的版本，把研究計畫盡量落地**」，能支撐你先完成實驗與報告；之後再把 Amazon Beauty 接進來即可。

## 檔案結構
- `preprocess_movielens.py`：資料預處理
- `dataset.py`：訓練 / 驗證 / 測試資料集
- `model.py`：CEINN 模型
- `train_ceinn.py`：訓練主程式
- `utils.py`：工具函式
- `config.yaml`：設定檔

## 1. 安裝
```bash
pip install -r requirements.txt
```

## 2. 預處理
請把 `movies.dat / ratings.dat / users.dat / README` 放在同一層。

```bash
python preprocess_movielens.py --data_dir ../ --out_dir ./outputs/preprocessed
```

## 3. 訓練
```bash
python train_ceinn.py --config config.yaml --ablation none
```

## 4. 消融實驗
```bash
# 無 Prospect Theory（基本效用評分）
python train_ceinn.py --config config.yaml --ablation w/o_pt

# 無 Hyperbolic Discounting（基本自注意力）
python train_ceinn.py --config config.yaml --ablation w/o_hd

# 無多任務學習（僅長期經濟信號）
python train_ceinn.py --config config.yaml --ablation w/o_mtl

# 無因果調整（無 SNIPS、無混淆因子）
python train_ceinn.py --config config.yaml --ablation w/o_causal
```

## 5. 輸出
會在 `outputs/` 底下生成：
- `best_model_*.pt`
- `metrics_*.json`
- `preprocessed/*.pkl`

## 建議先跑
若你怕 VSCode 第一次太慢，先把 `epochs` 改成 3。
