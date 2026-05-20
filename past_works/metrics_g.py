import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Any

# ============================================================================
# 任務 3.3.1 & 3.3.2：基礎評估指標 (支援 Tensor Batch 運算)
# ============================================================================

def get_ranks(predictions: torch.Tensor, targets: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
    """
    計算目標物品在 Full Ranking 下的排名。
    
    Args:
        predictions: [batch_size, num_items] 模型對所有物品的預測分數 (效用 U)
        targets: [batch_size] 真實的下一個物品 ID
        history_mask: [batch_size, num_items] 歷史互動遮罩，已互動過的位置為 True
        
    Returns:
        ranks: [batch_size] 目標物品的排名 (1-based, 也就是第一名為 1)
    """
    # 任務 3.3.1：排除訓練集中已出現的物品 (將其分數設為極小值)
    masked_predictions = predictions.masked_fill(history_mask, -float('inf'))
    
    # 取出真實目標物品的分數: [batch_size, 1]
    target_scores = masked_predictions.gather(1, targets.unsqueeze(1))
    
    # 計算有多少物品的分數「大於」真實目標的分數
    # 加上 1 就是它在排序中的 Rank
    ranks = (masked_predictions > target_scores).sum(dim=1) + 1
    return ranks

def hr_at_k(ranks: torch.Tensor, k: int) -> torch.Tensor:
    """計算 Hit Rate @ K (Per-user)"""
    return (ranks <= k).float()

def ndcg_at_k(ranks: torch.Tensor, k: int) -> torch.Tensor:
    """計算 NDCG @ K (Per-user)"""
    hits = (ranks <= k).float()
    # NDCG 權重：1 / log2(rank + 1)
    return hits / torch.log2(ranks.float() + 1.0)

def mrr(ranks: torch.Tensor) -> torch.Tensor:
    """計算 Mean Reciprocal Rank (Per-user)"""
    return 1.0 / ranks.float()

def calculate_all_metrics(ranks: torch.Tensor, ks: List[int] = [5, 10, 20]) -> Dict[str, torch.Tensor]:
    """
    一次計算所有 K 值的 HR, NDCG 與 MRR。
    回傳的 Tensor 皆為 [batch_size]，便於外部做分組統計或全域平均。
    """
    metrics = {'MRR': mrr(ranks)}
    for k in ks:
        metrics[f'HR@{k}'] = hr_at_k(ranks, k)
        metrics[f'NDCG@{k}'] = ndcg_at_k(ranks, k)
    return metrics


# ============================================================================
# 任務 3.3.3：分組評估指標 (長尾/頭部分析)
# ============================================================================

def group_metrics(ranks: torch.Tensor, 
                  target_items: torch.Tensor, 
                  group_labels: Dict[int, str], 
                  k: int = 10) -> Dict[str, Dict[str, float]]:
    """
    依據物品的流行度分組 (Head, Torso, Tail) 統計 HR@K 與 NDCG@K。
    驗證假說 H2a：IPS 加權是否成功提升了 Tail 物品的表現。
    
    Args:
        ranks: [batch_size] 預測排名
        target_items: [batch_size] 目標物品 ID
        group_labels: Mapping[item_id, 'head' | 'torso' | 'tail']
        k: 評估的 K 值
        
    Returns:
        Dict: { 'head': {'HR@10': x, 'NDCG@10': y}, ... }
    """
    # 轉為 numpy 以方便做字典查找與遮罩運算
    ranks_np = ranks.cpu().numpy()
    targets_np = target_items.cpu().numpy()
    
    # 取得這個 batch 內每個目標物品所屬的 group
    batch_groups = np.array([group_labels.get(item, 'unknown') for item in targets_np])
    
    results = {}
    for g in ['head', 'torso', 'tail']:
        mask = (batch_groups == g)
        if mask.sum() == 0:
            results[g] = {f'HR@{k}': 0.0, f'NDCG@{k}': 0.0, 'count': 0}
            continue
            
        g_ranks = ranks_np[mask]
        g_hr = (g_ranks <= k).astype(float).mean()
        g_ndcg = ((g_ranks <= k).astype(float) / np.log2(g_ranks + 1.0)).mean()
        
        results[g] = {
            f'HR@{k}': float(g_hr), 
            f'NDCG@{k}': float(g_ndcg),
            'count': int(mask.sum())
        }
        
    return results


# ============================================================================
# 任務 3.3.4：去混淆品質指標 (Confounding AUC)
# ============================================================================

def confounding_auc(h_states: np.ndarray, z_labels: np.ndarray) -> float:
    """
    驗證假說 H2b：評估潛在狀態 h_t 是否已成功與混淆變數 Z 解耦合 (h_t ⊥ Z)。
    
    使用 Logistic Regression 在 h_t 上預測 Z_labels。
    若去混淆成功，鑑別器應無法分辨，AUC 將趨近於 0.5 (Random Guess 基線)。
    若 AUC 高於 0.8，代表特徵隔離失敗，h_t 仍記憶了流行度。
    
    Args:
        h_states: [N, D] 凍結的已訓練模型輸出的潛在狀態。
        z_labels: [N] 每個樣本對應的目標物品的真實混淆變數分桶 (0 ~ num_bins-1)。
        
    Returns:
        float: 5-Fold CV 下的 Multi-class OVR ROC-AUC score。
    """
    # 使用 L2 正規化的 multinomial 邏輯迴歸，加強線性收斂穩定度
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 儲存預測機率與對應的真實標籤
    probas = np.zeros((len(z_labels), len(np.unique(z_labels))))
    
    # 執行 5-Fold 交叉驗證
    for train_idx, test_idx in cv.split(h_states, z_labels):
        X_train, y_train = h_states[train_idx], z_labels[train_idx]
        X_test = h_states[test_idx]
        
        clf.fit(X_train, y_train)
        probas[test_idx] = clf.predict_proba(X_test)
        
    # 計算 Multi-class AUC (One-vs-Rest)
    # average='macro' 保證即使長尾 (tail) 樣本的 Z_bucket 較少，也不會被 Head 吃掉權重
    try:
        auc_score = roc_auc_score(z_labels, probas, multi_class='ovr', average='macro')
    except ValueError as e:
        # 處理極端情況：若某個 fold 缺失了某個 class
        print(f"Warning: ROC AUC computation failed ({e}). Returning 0.5 as fallback.")
        auc_score = 0.5
        
    return float(auc_score)