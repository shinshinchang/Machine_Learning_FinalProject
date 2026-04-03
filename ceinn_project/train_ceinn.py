import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CEINNTrainDataset, CEINNEvalDataset
from model import AblationFlags, CEINN
from utils import ensure_dir, get_device, load_config, load_pickle, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--ablation', type=str, default='none', choices=['none', 'no_pt', 'no_hd', 'no_mtl', 'no_causal'])
    parser.add_argument('--preprocessed_path', type=str, default='./outputs/preprocessed/movielens_1m_preprocessed.pkl')
    return parser.parse_args()


def ablation_to_flags(name: str) -> AblationFlags:
    flags = AblationFlags(True, True, True, True)
    if name == 'no_pt':
        flags.use_pt = False
    elif name == 'no_hd':
        flags.use_hd = False
    elif name == 'no_mtl':
        flags.use_mtl = False
    elif name == 'no_causal':
        flags.use_causal = False
    return flags


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device)
    return out


def build_train_candidates(batch):
    pos_items = batch['pos_item'].unsqueeze(1)
    candidates = torch.cat([pos_items, batch['neg_items']], dim=1)
    short_labels = torch.cat([batch['pos_short_label'].unsqueeze(1), batch['neg_short_labels']], dim=1)
    long_labels = torch.cat([batch['pos_long_label'].unsqueeze(1), batch['neg_long_labels']], dim=1)
    return candidates, short_labels, long_labels


def compute_loss(outputs, short_labels, long_labels, loss_cfg, flags):
    weights = outputs['snips_weight'] if flags.use_causal else torch.ones_like(outputs['total_score'])

    short_loss = F.binary_cross_entropy_with_logits(outputs['short_score'], short_labels, reduction='none')
    long_loss = F.binary_cross_entropy_with_logits(outputs['long_score'], long_labels, reduction='none')
    total_loss = F.binary_cross_entropy_with_logits(outputs['total_score'], short_labels, reduction='none')

    short_loss = (short_loss * weights).mean()
    long_loss = (long_loss * weights).mean()
    total_loss = (total_loss * weights).mean()

    mtl_loss = short_loss + long_loss
    loss = total_loss + loss_cfg['lambda_ortho'] * outputs['ortho_reg']
    if flags.use_mtl:
        loss = loss + loss_cfg['lambda_mtl'] * mtl_loss
    return loss, {
        'loss_total': float(total_loss.detach().cpu()),
        'loss_short': float(short_loss.detach().cpu()),
        'loss_long': float(long_loss.detach().cpu()),
        'loss_ortho': float(outputs['ortho_reg'].detach().cpu()),
    }


@torch.no_grad()
def evaluate(model, loader, device, flags, k=10):
    model.eval()
    hits, ndcgs = [], []
    for batch in tqdm(loader, desc='Evaluating', leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch['seq'], batch['candidates'], flags)
        scores = outputs['total_score']
        _, rank_idx = torch.sort(scores, dim=1, descending=True)
        labels = batch['labels']
        for i in range(scores.size(0)):
            ranked_labels = labels[i][rank_idx[i][:k]]
            pos_positions = (ranked_labels == 1).nonzero(as_tuple=False)
            if len(pos_positions) > 0:
                rank = int(pos_positions[0].item())
                hits.append(1.0)
                ndcgs.append(1.0 / torch.log2(torch.tensor(rank + 2.0)).item())
            else:
                hits.append(0.0)
                ndcgs.append(0.0)
    return {'HR@10': sum(hits) / len(hits), 'NDCG@10': sum(ndcgs) / len(ndcgs)}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    flags = ablation_to_flags(args.ablation)
    set_seed(cfg['training']['seed'])
    device = get_device(cfg['training']['device'])

    payload = load_pickle(args.preprocessed_path)
    output_dir = Path(cfg['output']['save_dir'])
    ensure_dir(output_dir)

    train_ds = CEINNTrainDataset(
        payload['user_sequences'],
        payload['user_ratings'],
        payload['num_items'],
        payload['max_seq_len'],
        neg_samples=cfg['training']['neg_samples'],
    )
    valid_ds = CEINNEvalDataset(
        payload['user_sequences'],
        payload['user_ratings'],
        payload['num_items'],
        payload['max_seq_len'],
        mode='valid',
        num_eval_negatives=cfg['data']['num_eval_negatives'],
        seed=cfg['training']['seed'],
    )
    test_ds = CEINNEvalDataset(
        payload['user_sequences'],
        payload['user_ratings'],
        payload['num_items'],
        payload['max_seq_len'],
        mode='test',
        num_eval_negatives=cfg['data']['num_eval_negatives'],
        seed=cfg['training']['seed'] + 7,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
    )
    valid_loader = DataLoader(valid_ds, batch_size=cfg['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'], shuffle=False)

    model = CEINN(
        num_items=payload['num_items'],
        item_side_features=payload['item_side_features'],
        item_propensity=payload['item_propensity'],
        **cfg['model'],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )

    best_valid = -1.0
    best_metrics = None
    best_path = output_dir / f'best_model_{args.ablation}.pt'

    for epoch in range(1, cfg['training']['epochs'] + 1):
        model.train()
        running = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg["training"]["epochs"]}')
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            candidates, short_labels, long_labels = build_train_candidates(batch)
            outputs = model(batch['seq'], candidates, flags)
            loss, detail = compute_loss(outputs, short_labels, long_labels, cfg['loss'], flags)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            optimizer.step()

            running.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f'{sum(running)/len(running):.4f}')

        valid_metrics = evaluate(model, valid_loader, device, flags)
        print(f'\n[Epoch {epoch}] valid metrics: {valid_metrics}')
        if valid_metrics['NDCG@10'] > best_valid:
            best_valid = valid_metrics['NDCG@10']
            torch.save({'model_state_dict': model.state_dict(), 'config': cfg, 'ablation': args.ablation}, best_path)
            test_metrics = evaluate(model, test_loader, device, flags)
            best_metrics = {'valid': valid_metrics, 'test': test_metrics, 'epoch': epoch, 'ablation': args.ablation}
            print(f'[Epoch {epoch}] new best test metrics: {test_metrics}')

    metrics_path = output_dir / f'metrics_{args.ablation}.json'
    save_json(best_metrics, metrics_path)
    print(f'Best model saved to: {best_path}')
    print(f'Metrics saved to: {metrics_path}')
    print(best_metrics)


if __name__ == '__main__':
    main()
