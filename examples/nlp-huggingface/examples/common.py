"""Shared data/metrics helpers for the tiny teaching examples."""
from pathlib import Path
import csv
import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)

def reviews():
    with (Path(__file__).parent / 'reviews.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows: row['label'] = int(row['label'])
    return {s: [r for r in rows if r['split'] == s]
            for s in ('train','validation','test')}

def metrics(labels, predictions):
    from sklearn.metrics import accuracy_score, f1_score
    return {'accuracy': float(accuracy_score(labels, predictions)),
            'macro_f1': float(f1_score(labels, predictions, average='macro', zero_division=0))}
