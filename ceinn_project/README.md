# CEINN for MovieLens-1M (PyTorch)

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

## 重要說明
因為 MovieLens-1M 沒有「曝光紀錄」「加入收藏」「停留時間」「完整觀看率」這些欄位，所以研究計畫中的部分概念必須做工程近似：

1. **曝光 propensity**：用 item popularity 近似。
2. **長期價值標籤**：用較高評分（例如 >= 4）近似正向長期價值。
3. **短期誘惑標籤**：用一般互動事件（評分存在）近似。
4. **經濟屬性**：MovieLens 沒有價格，因此用 `genres / year / popularity` 當作 item side features。

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

## 4. 消融
```bash
python train_ceinn.py --config config.yaml --ablation no_pt
python train_ceinn.py --config config.yaml --ablation no_hd
python train_ceinn.py --config config.yaml --ablation no_mtl
python train_ceinn.py --config config.yaml --ablation no_causal
```

## 5. 輸出
會在 `outputs/` 底下生成：
- `best_model_*.pt`
- `metrics_*.json`
- `preprocessed/*.pkl`
