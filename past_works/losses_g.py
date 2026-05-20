import torch
import torch.nn as nn
import torch.nn.functional as F

class IPSWeightedChoiceLoss(nn.Module):
    """
    任務 3.2.1, 3.2.2, 3.2.4: 統一的 IPS 加權效用選擇損失函數。
    
    此模組同時兼具以下功能：
    1. 標準的序列交叉熵損失 L_seq (當 variant='none' 且輸入為純 ID logits 時)
    2. 效用選擇損失 L_C (當 variant='none' 且輸入為 Utilities 時)
    3. IPS 加權損失 L_IPS (當 variant='clipped' 或 'self_normalized' 時)
    """
    def __init__(self, variant: str = 'clipped', tau: float = 30.0, ignore_index: int = 0):
        super().__init__()
        valid_variants = ['none', 'clipped', 'self_normalized']
        if variant not in valid_variants:
            raise ValueError(f"IPS variant must be one of {valid_variants}, got {variant}")
        
        self.variant = variant
        self.tau = tau
        self.ignore_index = ignore_index

    def forward(self, utilities: torch.Tensor, targets: torch.Tensor, propensities: torch.Tensor = None) -> torch.Tensor:
        """
        計算 IPS 加權的負對數似然損失 (Negative Log-Likelihood)。
        
        Args:
            utilities: 形狀為 [batch_size, seq_len, num_items] 的預測效用值 (或原始 logits)。
            targets: 形狀為 [batch_size, seq_len] 的真實下一個互動物品 ID。
            propensities: 形狀為 [batch_size, seq_len] 的目標物品曝光傾向分數 p_i。
                          若 variant 為 'none'，此項可為 None。
                          
        Returns:
            torch.Tensor: 純量損失值 (已對 batch 內的有效樣本取平均)。
        """
        # 將資料攤平，方便使用 F.cross_entropy 計算
        # utilities_flat: [batch_size * seq_len, num_items]
        # targets_flat: [batch_size * seq_len]
        utilities_flat = utilities.view(-1, utilities.size(-1))
        targets_flat = targets.view(-1)
        
        # 找出非 PAD 的有效樣本遮罩
        valid_mask = (targets_flat != self.ignore_index)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=utilities.device, requires_grad=True)

        # 計算標準的無加權 Cross Entropy (L_C 或是 L_seq)
        # reduction='none' 讓我們可以針對每個樣本獨立取得 loss，以便後續乘上 IPS 權重
        ce_losses = F.cross_entropy(utilities_flat, targets_flat, reduction='none', ignore_index=self.ignore_index)
        
        # 僅提取有效樣本的損失
        valid_ce_losses = ce_losses[valid_mask]

        if self.variant == 'none' or propensities is None:
            # 消融實驗 A2 (w/o IPS) 或純觀測預測
            return valid_ce_losses.mean()

        # 處理 IPS 權重
        propensities_flat = propensities.view(-1)[valid_mask]
        
        # 為避免 p_i 極端趨近於 0 導致數值不穩定，加上微小 epsilon
        eps = 1e-6
        inv_propensities = 1.0 / (propensities_flat + eps)

        if self.variant == 'clipped':
            # 變異數縮減：Clipped IPS (w = min(1/p_i, tau))
            weights = torch.clamp(inv_propensities, max=self.tau)
        elif self.variant == 'self_normalized':
            # 變異數縮減：Self-Normalized IPS (w = (1/p_i) / sum(1/p_j))
            # 注意：這裡乘以 batch 內的有效樣本數 (len)，是為了保持梯度的量級與標準 CE 一致
            sn_sum = inv_propensities.sum() + eps
            weights = (inv_propensities / sn_sum) * len(inv_propensities)
        
        # 將權重不參與梯度計算 (detach)，僅作為損失放大的係數
        weights = weights.detach()

        # 最終 IPS 加權損失
        ips_loss = (valid_ce_losses * weights).mean()
        return ips_loss


class AdversarialLoss(nn.Module):
    """
    任務 3.2.3: 鑑別器對抗損失 L_adv
    
    鑑別器 D 試圖從去混淆潛在狀態 h_t 預測混淆變數 Z_i。
    注意：梯度反轉層 (Gradient Reversal Layer, GRL) 應在模型前向傳播時，
    於輸入至鑑別器之前的網路層介入。此處僅負責最純粹的 CE 計算。
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        # Z 分桶可能沒有 PAD 的概念，但若需忽略無效特徵，可傳入 ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, d_logits: torch.Tensor, z_targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d_logits: 形狀為 [batch_size * seq_len, num_z_buckets] 的鑑別器預測。
            z_targets: 形狀為 [batch_size * seq_len] 的真實 Z_i 分桶標籤。
        """
        # 如果有經過 padding 的序列，請在外部將無效的 logits 與 targets 過濾掉
        # 或透過 ignore_index 處理。
        return self.loss_fn(d_logits, z_targets)


class CombinedLoss(nn.Module):
    """
    任務 3.2.5: 聯合目標函數整合
    
    將主損失 (L_seq / L_C / L_IPS) 與對抗損失 (L_adv) 動態整合。
    L2 正則化 (L_reg) 建議直接交由 PyTorch Optimizer (如 AdamW 的 weight_decay) 處理，
    不需要在此處手動撰寫，以獲得最佳的底層 C++ 執行效能。
    """
    def __init__(self, lambda_adv: float = 0.1):
        super().__init__()
        self.lambda_adv = lambda_adv

    def forward(self, main_loss: torch.Tensor, adv_loss: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            main_loss: 來自 IPSWeightedChoiceLoss 的輸出。
            adv_loss: 來自 AdversarialLoss 的輸出 (若消融實驗關閉 GRL，則為 None)。
        """
        total_loss = main_loss
        
        if adv_loss is not None and self.lambda_adv > 0.0:
            total_loss = total_loss + self.lambda_adv * adv_loss
            
        return total_loss