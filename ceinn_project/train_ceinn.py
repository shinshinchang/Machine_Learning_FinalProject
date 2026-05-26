import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CEINNTrainDataset, CEINNEvalDataset
from model import AblationFlags, CEINN, GRU4Rec, SASRec
from utils import ensure_dir, get_device, load_config, load_pickle, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model', type=str, default='ceinn', choices=['ceinn', 'sasrec', 'gru4rec'])
    parser.add_argument('--ablation', type=str, default='none', choices=['none', 'w/o_pt', 'w/o_hd', 'w/o_mtl', 'w/o_causal'])
    parser.add_argument('--preprocessed_path', type=str, default='./outputs/preprocessed/movielens_1m_preprocessed.pkl')
    parser.add_argument('--popularity_analysis', action='store_true')
    return parser.parse_args()


def ablation_to_flags(name: str) -> AblationFlags:
    flags = AblationFlags(True, True, True, True)
    if name == 'w/o_pt':
        flags.use_pt = False
    elif name == 'w/o_hd':
        flags.use_hd = False
    elif name == 'w/o_mtl':
        flags.use_mtl = False
    elif name == 'w/o_causal':
        flags.use_causal = False
    return flags


def sanitize_ablation_name(name: str) -> str:
    """Convert ablation name to filesystem-safe format (w/o_pt -> w_o_pt)"""
    return name.replace('/', '_')


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


def build_model(model_name, payload, cfg):
    if model_name == 'ceinn':
        return CEINN(
            num_items=payload['num_items'],
            item_side_features=payload['item_side_features'],
            item_propensity=payload['item_propensity'],
            **cfg['model'],
        )
    if model_name == 'sasrec':
        return SASRec(
            num_items=payload['num_items'],
            max_seq_len=payload['max_seq_len'],
            **cfg['model'],
        )
    if model_name == 'gru4rec':
        return GRU4Rec(
            num_items=payload['num_items'],
            **cfg['model'],
        )
    raise ValueError(f'Unsupported model: {model_name}')


def forward_model(model, seq, candidates, model_name, flags):
    if model_name == 'ceinn':
        return model(seq, candidates, flags)
    return model(seq, candidates)


def compute_loss(outputs, short_labels, long_labels, loss_cfg, flags, model_name):
    if model_name != 'ceinn':
        total_loss = F.binary_cross_entropy_with_logits(outputs['total_score'], short_labels)
        return total_loss, {
            'loss_total': float(total_loss.detach().cpu()),
            'loss_short': 0.0,
            'loss_long': 0.0,
            'loss_ortho': 0.0,
        }

    weights = outputs['snips_weight'] if flags.use_causal else torch.ones_like(outputs['total_score'])
    total_loss = F.binary_cross_entropy_with_logits(outputs['total_score'], short_labels, reduction='none')
    total_loss = (total_loss * weights).mean()

    ortho_reg = outputs['ortho_reg']
    loss = total_loss + loss_cfg['lambda_ortho'] * ortho_reg

    short_loss_value = 0.0
    long_loss_value = 0.0
    if flags.use_mtl:
        short_loss = F.binary_cross_entropy_with_logits(outputs['short_score'], short_labels, reduction='none')
        long_loss = F.binary_cross_entropy_with_logits(outputs['long_score'], long_labels, reduction='none')
        short_loss = (short_loss * weights).mean()
        long_loss = (long_loss * weights).mean()
        loss = loss + loss_cfg['lambda_mtl'] * (short_loss + long_loss)
        short_loss_value = float(short_loss.detach().cpu())
        long_loss_value = float(long_loss.detach().cpu())

    return loss, {
        'loss_total': float(total_loss.detach().cpu()),
        'loss_short': short_loss_value,
        'loss_long': long_loss_value,
        'loss_ortho': float(ortho_reg.detach().cpu()),
    }


def build_popularity_groups(train_ds, num_items):
    counts = torch.zeros(num_items + 1, dtype=torch.long)
    for _, _, pos_item, _ in train_ds.samples:
        counts[pos_item] += 1

    item_ids = list(range(1, num_items + 1))
    item_ids.sort(key=lambda x: (int(counts[x]), x), reverse=True)

    n_items = len(item_ids)
    head_n = max(1, int(n_items * 0.2))
    tail_n = max(1, int(n_items * 0.2))
    mid_end = max(head_n, n_items - tail_n)

    head_items = set(item_ids[:head_n])
    mid_items = set(item_ids[head_n:mid_end])
    tail_items = set(item_ids[mid_end:])

    return {
        'head': head_items,
        'mid': mid_items,
        'tail': tail_items,
    }


def get_popularity_bucket(item_id, popularity_groups):
    if item_id in popularity_groups['head']:
        return 'head'
    if item_id in popularity_groups['mid']:
        return 'mid'
    return 'tail'


@torch.no_grad()
def evaluate(model, loader, device, model_name, flags, k=10, popularity_groups=None):
    model.eval()
    hits, ndcgs = [], []
    bucket_hits = defaultdict(list)
    bucket_ndcgs = defaultdict(list)
    for batch in tqdm(loader, desc='Evaluating', leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = forward_model(model, batch['seq'], batch['candidates'], model_name, flags)
        scores = outputs['total_score']
        _, rank_idx = torch.sort(scores, dim=1, descending=True)
        labels = batch['labels']
        for i in range(scores.size(0)):
            ranked_labels = labels[i][rank_idx[i][:k]]
            pos_positions = (ranked_labels == 1).nonzero(as_tuple=False)
            hit_val = 0.0
            ndcg_val = 0.0
            if len(pos_positions) > 0:
                rank = int(pos_positions[0].item())
                hit_val = 1.0
                ndcg_val = 1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
            hits.append(hit_val)
            ndcgs.append(ndcg_val)

            if popularity_groups is not None:
                pos_item = int(batch['candidates'][i, 0].item())
                bucket = get_popularity_bucket(pos_item, popularity_groups)
                bucket_hits[bucket].append(hit_val)
                bucket_ndcgs[bucket].append(ndcg_val)

    metrics = {'HR@10': sum(hits) / len(hits), 'NDCG@10': sum(ndcgs) / len(ndcgs)}
    if popularity_groups is not None:
        pop_metrics = {}
        for bucket in ['head', 'mid', 'tail']:
            b_hits = bucket_hits[bucket]
            b_ndcgs = bucket_ndcgs[bucket]
            if len(b_hits) == 0:
                pop_metrics[bucket] = {'HR@10': 0.0, 'NDCG@10': 0.0, 'count': 0}
            else:
                pop_metrics[bucket] = {
                    'HR@10': sum(b_hits) / len(b_hits),
                    'NDCG@10': sum(b_ndcgs) / len(b_ndcgs),
                    'count': len(b_hits),
                }
        metrics['popularity_analysis'] = pop_metrics
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    effective_ablation = args.ablation if args.model == 'ceinn' else 'none'
    flags = ablation_to_flags(effective_ablation)

    if args.model != 'ceinn' and args.ablation != 'none':
        print(f'[Info] --ablation is ignored for model={args.model}; using ablation=none')

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

    popularity_groups = None
    if args.popularity_analysis:
        popularity_groups = build_popularity_groups(train_ds, payload['num_items'])

    model = build_model(args.model, payload, cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )

    best_valid = -1.0
    best_metrics = None
    if args.model == 'ceinn':
        name_suffix = sanitize_ablation_name(effective_ablation)
    else:
        name_suffix = args.model
    best_path = output_dir / f'best_model_{name_suffix}.pt'

    for epoch in range(1, cfg['training']['epochs'] + 1):
        model.train()
        running = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg["training"]["epochs"]}')
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            candidates, short_labels, long_labels = build_train_candidates(batch)
            outputs = forward_model(model, batch['seq'], candidates, args.model, flags)
            loss, detail = compute_loss(outputs, short_labels, long_labels, cfg['loss'], flags, args.model)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            optimizer.step()

            running.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f'{sum(running)/len(running):.4f}')

        valid_metrics = evaluate(model, valid_loader, device, args.model, flags)
        print(f'\n[Epoch {epoch}] valid metrics: {valid_metrics}')
        if valid_metrics['NDCG@10'] > best_valid:
            best_valid = valid_metrics['NDCG@10']
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'config': cfg,
                    'ablation': effective_ablation,
                    'model': args.model,
                },
                best_path,
            )
            test_metrics = evaluate(
                model,
                test_loader,
                device,
                args.model,
                flags,
                popularity_groups=popularity_groups,
            )
            best_metrics = {
                'model': args.model,
                'ablation': effective_ablation,
                'valid': valid_metrics,
                'test': test_metrics,
                'epoch': epoch,
            }
            if args.popularity_analysis and 'popularity_analysis' in test_metrics:
                best_metrics['popularity_analysis'] = test_metrics['popularity_analysis']
            print(f'[Epoch {epoch}] new best test metrics: {test_metrics}')

    metrics_path = output_dir / f'metrics_{name_suffix}.json'
    save_json(best_metrics, metrics_path)
    print(f'Best model saved to: {best_path}')
    print(f'Metrics saved to: {metrics_path}')
    print(best_metrics)


if __name__ == '__main__':
    main()
