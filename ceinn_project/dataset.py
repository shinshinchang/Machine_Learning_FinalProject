import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import sequence_padding


class CEINNTrainDataset(Dataset):
    def __init__(
        self,
        user_sequences: Dict[int, List[int]],
        user_ratings: Dict[int, List[float]],
        num_items: int,
        max_seq_len: int,
        neg_samples: int = 5,
        long_term_threshold: float = 4.0,
    ):
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.neg_samples = neg_samples
        self.long_term_threshold = long_term_threshold
        self.samples = []
        self.user_histories = {u: set(seq) for u, seq in user_sequences.items()}

        for u in sorted(user_sequences.keys()):
            seq = user_sequences[u]
            rs = user_ratings[u]
            # reserve last 2 for valid/test
            if len(seq) < 4:
                continue
            for t in range(1, len(seq) - 2):
                hist = seq[:t]
                target = seq[t]
                rating = rs[t]
                self.samples.append((u, hist, target, rating))

    def __len__(self):
        return len(self.samples)

    def _sample_negative(self, user_id: int):
        user_items = self.user_histories[user_id]
        while True:
            item = random.randint(1, self.num_items)
            if item not in user_items:
                return item

    def __getitem__(self, idx):
        user_id, hist, pos_item, rating = self.samples[idx]
        seq = sequence_padding(hist, self.max_seq_len, pad_value=0)
        pos_long = 1.0 if rating >= self.long_term_threshold else 0.0

        neg_items = [self._sample_negative(user_id) for _ in range(self.neg_samples)]
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'seq': torch.tensor(seq, dtype=torch.long),
            'pos_item': torch.tensor(pos_item, dtype=torch.long),
            'pos_short_label': torch.tensor(1.0, dtype=torch.float32),
            'pos_long_label': torch.tensor(pos_long, dtype=torch.float32),
            'neg_items': torch.tensor(neg_items, dtype=torch.long),
            'neg_short_labels': torch.zeros(self.neg_samples, dtype=torch.float32),
            'neg_long_labels': torch.zeros(self.neg_samples, dtype=torch.float32),
        }


class CEINNEvalDataset(Dataset):
    def __init__(
        self,
        user_sequences: Dict[int, List[int]],
        user_ratings: Dict[int, List[float]],
        num_items: int,
        max_seq_len: int,
        mode: str = 'valid',
        num_eval_negatives: int = 100,
        seed: int = 42,
    ):
        assert mode in {'valid', 'test'}
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.num_eval_negatives = num_eval_negatives
        self.rng = random.Random(seed)
        self.samples = []
        self.user_histories = {u: set(seq) for u, seq in user_sequences.items()}

        for u in sorted(user_sequences.keys()):
            seq = user_sequences[u]
            rs = user_ratings[u]
            if len(seq) < 3:
                continue
            if mode == 'valid':
                hist = seq[:-2]
                target = seq[-2]
                rating = rs[-2]
            else:
                hist = seq[:-1]
                target = seq[-1]
                rating = rs[-1]
            self.samples.append((u, hist, target, rating))

    def __len__(self):
        return len(self.samples)

    def _sample_negatives(self, user_id: int):
        user_items = self.user_histories[user_id]
        negatives = set()
        while len(negatives) < self.num_eval_negatives:
            item = self.rng.randint(1, self.num_items)
            if item not in user_items:
                negatives.add(item)
        return list(negatives)

    def __getitem__(self, idx):
        user_id, hist, pos_item, rating = self.samples[idx]
        seq = sequence_padding(hist, self.max_seq_len, pad_value=0)
        candidates = [pos_item] + self._sample_negatives(user_id)
        labels = [1] + [0] * (len(candidates) - 1)
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'seq': torch.tensor(seq, dtype=torch.long),
            'candidates': torch.tensor(candidates, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'target_rating': torch.tensor(rating, dtype=torch.float32),
        }
