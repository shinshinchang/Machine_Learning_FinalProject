"""
math_utils.py

CEINN (Causal Economics-Informed Neural Networks) 的底層統計與數學運算模組。
負責：
1. Log-Quantile 分桶 (針對時間間隔 Δt 與混淆變數 Z，解決重尾分布問題)
2. 集合相似度運算 (Standard Jaccard & IDF-weighted Jaccard，用於行為摩擦力 GenreRed)
3. 模組載入時的自我驗證測實驗證 (_self_check)

所有計算皆基於 NumPy，不依賴深度學習框架，以確保在 Phase 2 前處理階段可快速執行。
"""

import numpy as np
from typing import List, Tuple, Optional

# ============================================================================
# 1. 對數分位數分桶 (Log-Quantile Bucketing)
# ============================================================================

def fit_log_quantile_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    """
    在對數空間中擬合等頻 (Quantile) 分桶的邊界。
    主要用於處理跨越多個數量級的重尾分布特徵（如相鄰互動時間差 Δt 或 SalesRank）。
    
    為避免未來資訊洩漏，此函式應僅在「訓練集」的資料上呼叫，
    取得 edges 後將其凍結，並應用於驗證與測試集。

    Args:
        values (np.ndarray): 訓練集中的數值特徵矩陣 (需為非負數)。
        num_bins (int): 預期的真實分桶數量 (不含保留的 PAD bucket)。

    Returns:
        np.ndarray: 單調遞增的邊界值陣列。
    """
    # 過濾無效值並進行對數平滑轉換 log(x + 1)
    valid_values = values[~np.isnan(values)]
    valid_values = np.clip(valid_values, a_min=0.0, a_max=None)
    log_vals = np.log1p(valid_values)
    
    # 計算等分位數點
    quantiles = np.linspace(0, 1, num_bins + 1)
    edges_log = np.quantile(log_vals, quantiles)
    
    # 轉換回原始空間 (exp(x) - 1)
    edges = np.expm1(edges_log)
    
    # 確保首尾包含所有可能的未來數值
    edges[0] = -np.inf
    edges[-1] = np.inf
    
    # 處理可能因分布過度集中產生的重複邊界 (如大量的 0)
    unique_edges = np.unique(edges)
    return np.sort(unique_edges)

def apply_bucketization(values: np.ndarray, edges: np.ndarray, reserve_pad: bool = True) -> np.ndarray:
    """
    將連續數值映射至預先擬合好的分桶中。
    
    Args:
        values (np.ndarray): 欲分桶的目標陣列。
        edges (np.ndarray): 由 `fit_log_quantile_edges` 產出的邊界陣列。
        reserve_pad (bool): 若為 True，所有產出的 index 會向右平移 1，將 index 0 保留給 PAD。

    Returns:
        np.ndarray: 整數型態的分桶 Index 陣列。
    """
    # np.digitize 預設會將值分佈至 1 ~ len(edges)-1
    # 若 reserve_pad = True，我們不需額外減 1，因為 0 會自然空出來做為 PAD
    bin_indices = np.digitize(values, edges)
    
    if not reserve_pad:
        bin_indices = bin_indices - 1
        
    return bin_indices


# ============================================================================
# 2. 集合相似度與行為摩擦力運算 (Behavioral Friction & GenreRed)
# ============================================================================

def compute_genre_idf(genre_matrix: np.ndarray) -> np.ndarray:
    """
    計算物品集合的類型逆文件頻率 (IDF)。
    此運算必須在 5-core 過濾後的「全電影集合」上進行，以反映生態圈的全局稀有度。
    
    數學定義: IDF_j = log(N / (DF_j + 1))
    
    Args:
        genre_matrix (np.ndarray): 尺寸為 (num_items, num_genres) 的二進位矩陣。
        
    Returns:
        np.ndarray: 尺寸為 (num_genres,) 的 IDF 權重陣列。
    """
    num_items = genre_matrix.shape[0]
    # 計算每個類型的出現次數 (Document Frequency)
    df = np.sum(genre_matrix, axis=0)
    
    # 平滑化 IDF 避免除以零
    idf = np.log(num_items / (df + 1.0))
    return idf

def compute_jaccard(history_union: np.ndarray, candidate: np.ndarray) -> float:
    """
    計算標準 Jaccard 相似度。
    用於衡量候選物品類型與使用者近期歷史類型集合的重疊度。
    
    Args:
        history_union (np.ndarray): 使用者歷史累積之類型二進位向量 (1D)。
        candidate (np.ndarray): 候選物品之類型二進位向量 (1D)。
        
    Returns:
        float: 相似度分數 [0.0, 1.0]
    """
    intersection = np.sum(np.logical_and(history_union, candidate))
    union = np.sum(np.logical_or(history_union, candidate))
    
    if union == 0:
        return 0.0
    return float(intersection / union)

def compute_idf_weighted_jaccard(history_union: np.ndarray, 
                                 candidate: np.ndarray, 
                                 idf_weights: np.ndarray) -> float:
    """
    計算 IDF 加權的 Jaccard 相似度。
    削弱如 Drama / Comedy 等高頻主流類型的權重，強調小眾類型的重疊，
    提供更精確的「選擇摩擦力 / 疲勞度」估計。
    
    數學定義: sum(IDF_{intersect}) / sum(IDF_{union})
    
    Args:
        history_union (np.ndarray): 使用者歷史累積之類型二進位向量 (1D)。
        candidate (np.ndarray): 候選物品之類型二進位向量 (1D)。
        idf_weights (np.ndarray): 預先計算好的類型 IDF 陣列。
        
    Returns:
        float: 加權相似度分數 [0.0, 1.0]
    """
    intersection_mask = np.logical_and(history_union, candidate)
    union_mask = np.logical_or(history_union, candidate)
    
    intersection_score = np.sum(idf_weights[intersection_mask])
    union_score = np.sum(idf_weights[union_mask])
    
    if union_score <= 0.0:
        return 0.0
    return float(intersection_score / union_score)


# ============================================================================
# 3. 模組自我驗證 (Self-Check Invariants)
# ============================================================================

def _self_check():
    """
    於模組載入時自動執行的不變量 (Invariants) 檢查。
    驗證邊界分桶邏輯、保留 PAD 的指標平移、以及 IDF 加權 Jaccard 計算的正確性。
    """
    # --- Check 1: Log-Quantile Bucketing ---
    # 建立一個包含極端長尾的虛擬數列 (類似 salesRank)
    test_vals = np.array([0, 1, 5, 10, 100, 1000, 10000, 100000, 500000, 1000000])
    edges = fit_log_quantile_edges(test_vals, num_bins=4)
    
    # 驗證 PAD 保留機制 (reserve_pad=True 時，不會出現 index 0)
    bucketed = apply_bucketization(test_vals, edges, reserve_pad=True)
    assert np.min(bucketed) >= 1, "Bucketing failed to reserve PAD index 0."
    assert np.max(bucketed) <= len(edges), "Bucketing indices out of expected range."
    
    # --- Check 2: Jaccard & IDF calculation ---
    # 假設有 3 個 item, 4 個 genre 的生態圈
    # genre 0 出現 3 次 (極熱門), genre 3 出現 1 次 (極冷門)
    genre_mat = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1]
    ])
    idf = compute_genre_idf(genre_mat)
    
    # 驗證稀有類型的 IDF > 熱門類型的 IDF
    assert idf[3] > idf[0], "IDF computation inverted: rare genre should have higher weight."
    
    # --- Check 3: Weighted Jaccard ---
    hist = np.array([1, 1, 0, 0])      # 包含了熱門的 genre 0 與 genre 1
    cand1 = np.array([1, 0, 0, 0])     # 只中了熱門 genre 0
    cand2 = np.array([0, 1, 0, 0])     # 只中了 genre 1 (相對較冷門)
    
    jaccard1 = compute_jaccard(hist, cand1)
    jaccard2 = compute_jaccard(hist, cand2)
    # 標準 Jaccard 下，兩者重疊度均為 1/2
    assert np.isclose(jaccard1, 0.5), "Standard Jaccard computation error."
    assert np.isclose(jaccard2, 0.5), "Standard Jaccard computation error."
    
    idf_jac1 = compute_idf_weighted_jaccard(hist, cand1, idf)
    idf_jac2 = compute_idf_weighted_jaccard(hist, cand2, idf)
    # IDF 加權下，命中較冷門類型 (cand2) 的相似度佔比應高於命中極熱門類型 (cand1)
    assert idf_jac2 > idf_jac1, "IDF weighted Jaccard failed to amplify rare intersections."

# 執行自檢
_self_check()