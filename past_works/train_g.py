"""
Phase 6: CEINN Training Script (train.py) - Low RAM Optimized & Loader Fixed
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import gc
import warnings

# 隱藏 NumPy 底層讀取 pickle 時的 DeprecationWarning，保持終端機乾淨
warnings.filterwarnings("ignore", category=DeprecationWarning)

from models.ceinn import build_ceinn_amazon, build_ceinn_movielens
from models.causal_deconfounder import alpha_schedule
from utils.losses import IPSLoss, AdversarialCE, UtilityChoiceLoss, CombinedLoss
from utils.metrics import compute_full_ranking, ndcg_at_k

# =============================================================================
# Loader 與 Dataset 準備區段
# =============================================================================
def load_backend_loader(dataset_name: str, data_dir: str):
    """
    正確的載入方式：直接呼叫 Loader 內建的 classmethod from_directory()，
    讓 Loader 自行處理複雜的 pickle 與 json 反序列化。
    """
    print(f"[*] 正在從 {data_dir} 載入記憶體 (Low RAM 環境請稍候)...")
    if dataset_name == "amazon":
        from data_loaders.amazon_beauty_loader import AmazonBeautyLoader
        return AmazonBeautyLoader.from_directory(data_dir)
    else:
        from data_loaders.movieslens_10M_loader import MovieLens10MLoader
        return MovieLens10MLoader.from_directory(data_dir)

class CEINN_DefaultDataset(Dataset):
    """
    通用型後備 Dataset。將使用者訓練序列展開為多個 Prefix 以進行 Next-item prediction。
    """
    def __init__(self, loader, split="train"):
        self.loader = loader
        self.split = split
        self.max_len = int(loader.vocab["max_seq_len"])
        self.pad_idx = int(loader.vocab["pad_index"])
        self.samples = []
        self.dataset_name = "amazon" if hasattr(loader, "item_meta") else "movielens"

        seqs = loader.train_seqs if split == "train" else (loader.val_seqs if split == "val" else loader.test_seqs)
        
        if split == "train":
            for u, seq in seqs.items():
                if len(seq) < 2: continue
                for i in range(1, len(seq)):
                    self.samples.append((u, i))
        else:
            for u, tgt in seqs.items():
                inp_seq = loader.train_seqs.get(u, [])
                if len(inp_seq) == 0: continue
                self.samples.append((u, -1, tgt))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        if self.split == "train":
            u, i = self.samples[idx]
            seq = self.loader.train_seqs[u]
            inp = seq[max(0, i - self.max_len):i]
            tgt = seq[i]
        else:
            u, _, tgt = self.samples[idx]
            inp = self.loader.train_seqs[u][-self.max_len:]

        # 左側 Padding
        pad_len = self.max_len - len(inp)
        padded_inp = [(self.pad_idx, self.pad_idx, self.pad_idx)] * pad_len + inp
        
        item_ids = torch.tensor([x[0] for x in padded_inp], dtype=torch.long)
        rating_ids = torch.tensor([x[1] for x in padded_inp], dtype=torch.long)
        dt_ids = torch.tensor([x[2] for x in padded_inp], dtype=torch.long)
        target_item = torch.tensor(tgt[0], dtype=torch.long)

        batch = {
            "item_ids": item_ids, "rating_ids": rating_ids, "dt_ids": dt_ids,
            "target_item": target_item
        }

        # 針對兩種 Dataset 配置 Target 的 Z Bucket 與特徵矩陣
        if self.dataset_name == "amazon":
            batch["z_buckets_target"] = torch.tensor(self.loader.Z_of(tgt[0]), dtype=torch.long)
        else:
            # MovieLens 需要動態撈取 Z 與預先分配 (V_cat,) 空間供 economics_utility.py 計算 Cost
            n_items = self.loader.vocab["n_items"]
            V_cat = n_items + 1
            
            if self.split == "train":
                batch["z_buckets_target"] = torch.tensor(self.loader.Z_at(u, i), dtype=torch.long)
            else:
                batch["z_buckets_target"] = torch.tensor(0, dtype=torch.long) 
                
            # 建立 (V_cat,) 大小的 Float Tensor 供 MovieLensCost 計算
            # 未來可在這裡或 training loop 中，透過 batch_jaccard 即時將全目錄的 GenreRed 寫入
            batch["genre_red"] = torch.zeros(V_cat, dtype=torch.float32)
            batch["recency_press"] = torch.zeros(V_cat, dtype=torch.float32)
            batch["pop_press"] = torch.zeros(V_cat, dtype=torch.float32)
            
            # 若為 training，精確填入 Target Item 的 Precomputed 值
            if self.split == "train":
                target_idx = tgt[0]
                batch["genre_red"][target_idx] = self.loader.genre_red_at(u, i)
        
        return batch

# =============================================================================
# 訓練邏輯
# =============================================================================
def train_propensity_warmup(model, dataloader, optimizer, epochs, device):
    """Propensity Estimator 預熱"""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    print(f"[*] 開始 Propensity Estimator 預熱 ({epochs} Epochs)...")
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            z_pos = batch['z_buckets_target'].to(device)
            logits_pos = model.deconfounder.propensity_logit(z_pos)
            
            z_neg = torch.randint(0, model.n_z_buckets, size=z_pos.shape, device=device)
            logits_neg = model.deconfounder.propensity_logit(z_neg)
            
            logits = torch.cat([logits_pos, logits_neg])
            labels = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)])
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  - Warmup Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["amazon", "movielens"], required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["training"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    # 1. 載入 Loader 與模型
    loader_backend = load_backend_loader(args.dataset, cfg["preprocess"]["output_dir"])
    
    arch_cfg = train_cfg["architecture"].copy()

    if args.dataset == "amazon":
        model = build_ceinn_amazon(loader_backend, **arch_cfg).to(device)
    else:
        # MovieLens 不允許 propensity_z_emb_dim，將其剔除
        if 'propensity_z_emb_dim' in arch_cfg:
            del arch_cfg['propensity_z_emb_dim']
        model = build_ceinn_movielens(loader_backend, n_z_buckets=train_cfg["n_z_buckets"], **arch_cfg).to(device)

    # 建立 DataLoader (具備 Fallback 機制)
    if hasattr(loader_backend, "get_dataloader"):
        train_loader = loader_backend.get_dataloader("train", train_cfg["batch_size"])
        val_loader = loader_backend.get_dataloader("val", train_cfg["val_batch_size"])
    else:
        train_loader = DataLoader(CEINN_DefaultDataset(loader_backend, "train"), batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg["num_workers"])
        val_loader = DataLoader(CEINN_DefaultDataset(loader_backend, "val"), batch_size=train_cfg["val_batch_size"], shuffle=False, num_workers=train_cfg["num_workers"])

    # 2. 定義 Losses 與 Optimizers
    criterion_ips = IPSLoss(
        variant=train_cfg["ips_variant"], clip_tau=train_cfg["ips_clip_tau"], 
        pad_index=loader_backend.vocab["pad_index"]
    )
    criterion_adv = AdversarialCE()
    criterion_combined = CombinedLoss(lambda_adv=train_cfg["lambda_adv"], lambda_reg=train_cfg["lambda_reg"])

    main_params = list(model.backbone.parameters()) + list(model.economics.parameters())
    opt_main = torch.optim.Adam(main_params, lr=train_cfg["lr"], weight_decay=train_cfg["lambda_reg"])
    opt_disc = torch.optim.Adam(model.deconfounder.discriminator.parameters(), lr=train_cfg["lr"])
    opt_prop = torch.optim.Adam(model.deconfounder.propensity_estimator.parameters(), lr=train_cfg["propensity_lr"])

    if train_cfg["propensity_warmup_epochs"] > 0:
        train_propensity_warmup(model, train_loader, opt_prop, train_cfg["propensity_warmup_epochs"], device)

    # 3. 聯合訓練主迴圈
    best_val_ndcg = -1.0
    patience = 0
    accum_steps = train_cfg.get("accum_steps", 1)  # 梯度累積

    print(f"[*] 開始 CEINN 聯合訓練 (Batch: {train_cfg['batch_size']}, Accum: {accum_steps})")
    for epoch in range(train_cfg["max_epochs"]):
        model.train()
        alpha = alpha_schedule(epoch / train_cfg["max_epochs"])
        
        ep_losses = {"total": 0.0, "ips": 0.0, "adv": 0.0}
        disc_correct, disc_total = 0, 0

        opt_main.zero_grad()
        opt_disc.zero_grad()

        for step, batch in enumerate(train_loader):
            # 將 Batch 移至 GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # ----------------------------------------------------
            # Step 1: Backbone + Utility + GRL Adversarial 
            # ----------------------------------------------------
            out = model(batch, grl_alpha=alpha)
            
            loss_choice = criterion_ips(out['U'], batch['target_item'], out['propensity_target'])
            loss_adv = criterion_adv(out['z_logits'], batch['z_buckets_target'])
            
            # 使用 L2 Norm 來作為正規化計算（從 Optimizer 接手這項會更有效率，但 CombinedLoss 仍可處理）
            loss_dict = criterion_combined(ips=loss_choice, adv=loss_adv, choice=torch.tensor(0.0).to(device))
            loss_main = loss_dict['total'] / accum_steps
            
            # GRL 反向傳播 (梯度會反號流回 Backbone)
            loss_main.backward()

            # 統計 Disc 準確率
            preds = out['z_logits'].argmax(dim=1)
            disc_correct += (preds == batch['z_buckets_target']).sum().item()
            disc_total += preds.size(0)

            # ----------------------------------------------------
            # Step 2: 獨立更新 Discriminator
            # ----------------------------------------------------
            h_t_detached = out['h_t'].detach() # 阻斷梯度回流至 Backbone
            z_logits_disc = model.deconfounder.discriminate(h_t_detached, alpha=0.0)
            loss_disc = criterion_adv(z_logits_disc, batch['z_buckets_target']) / accum_steps
            loss_disc.backward()

            ep_losses["total"] += loss_dict['total'].item()
            ep_losses["ips"] += loss_dict['L_IPS'].item()
            ep_losses["adv"] += loss_dict['L_adv'].item()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(main_params, train_cfg["grad_clip"])
                torch.nn.utils.clip_grad_norm_(model.deconfounder.discriminator.parameters(), train_cfg["grad_clip"])
                opt_main.step()
                opt_disc.step()
                opt_main.zero_grad()
                opt_disc.zero_grad()

            # Low RAM 極致優化：提早刪除圖節點釋放記憶體
            del out, loss_main, loss_choice, loss_adv, loss_dict, loss_disc, h_t_detached, z_logits_disc
            if step % 500 == 0: gc.collect() 

        num_batches = len(train_loader)
        print(f"Epoch {epoch:03d} | L_tot: {ep_losses['total']/num_batches:.4f} | "
              f"L_IPS: {ep_losses['ips']/num_batches:.4f} | L_adv: {ep_losses['adv']/num_batches:.4f} | "
              f"Disc_Acc: {disc_correct/disc_total:.4f} | alpha: {alpha:.4f}")

        # ----------------------------------------------------
        # Evaluation (任務 6.1.4: Early Stopping)
        # ----------------------------------------------------
        model.eval()
        val_ndcg_list = []
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                val_out = model(val_batch, grl_alpha=0.0)
                scores = val_out['U'].cpu().numpy()
                targets = val_batch['target_item'].cpu().numpy()
                
                for i in range(scores.shape[0]):
                    rank = compute_full_ranking(scores[i], targets[i], pad_index=loader_backend.vocab["pad_index"])
                    val_ndcg_list.append(ndcg_at_k([rank], k=10)[0])
                    
        val_ndcg10 = np.mean(val_ndcg_list)
        print(f" ---> Val NDCG@10: {val_ndcg10:.4f}")
        
        if val_ndcg10 > best_val_ndcg:
            best_val_ndcg = val_ndcg10
            patience = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience += 1
            if patience >= train_cfg["early_stop_patience"]:
                print(f"[*] Early stopping triggered. Best Val NDCG@10: {best_val_ndcg:.4f}")
                break

if __name__ == "__main__":
    main()