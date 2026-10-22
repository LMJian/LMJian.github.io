"""One training loop for RNN, LSTM and GRU, with correct variable-length handling."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from common import seed_everything, reviews, metrics

class ReviewDataset(Dataset):
    def __init__(self, rows, vocab):
        self.rows, self.vocab = rows, vocab
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        ids = [self.vocab.get(c,1) for c in row['text'][:64]] or [1]
        return torch.tensor(ids), float(row['label'])

def collate(batch):
    seqs, labels = zip(*batch)
    return (pad_sequence(seqs, batch_first=True, padding_value=0),
            torch.tensor([len(x) for x in seqs]), torch.tensor(labels))

class Classifier(nn.Module):
    def __init__(self, vocab_size, arch):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 24, padding_idx=0)
        self.rnn = {'rnn':nn.RNN, 'lstm':nn.LSTM, 'gru':nn.GRU}[arch](
            24, 32, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(64, 1)
    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.emb(ids), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, state = self.rnn(packed)
        h = state[0] if isinstance(state, tuple) else state
        return self.head(torch.cat([h[-2], h[-1]], dim=-1)).squeeze(-1)

@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    labels, predictions, loss_sum, n = [], [], 0., 0
    for ids, lengths, target in loader:
        logits = model(ids, lengths)
        loss_sum += nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='sum').item()
        n += len(target)
        labels.extend(target.int().tolist())
        predictions.extend((logits.sigmoid() >= 0.5).int().tolist())
    return loss_sum/n, metrics(labels, predictions)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['rnn','lstm','gru'], default='gru')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--out', default='runs/sequence')
    args = p.parse_args()
    if args.epochs < 1: p.error('--epochs must be positive')
    seed_everything()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = reviews()
    chars = sorted(set(''.join(r['text'] for r in data['train'])))
    vocab = {c:i for i,c in enumerate(['<pad>','<unk>']+chars)}
    loaders = {s:DataLoader(ReviewDataset(rows,vocab), batch_size=8,
                           shuffle=s=='train', collate_fn=collate)
               for s,rows in data.items()}
    model = Classifier(len(vocab), args.arch)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    best = float('inf')
    for epoch in range(args.epochs):
        model.train()
        for ids, lengths, target in loaders['train']:
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(ids,lengths), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_loss, _ = evaluate(model, loaders['validation'])
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), out/'model.pt')
    (out/'vocab.json').write_text(json.dumps(vocab,ensure_ascii=False), encoding='utf-8')
    (out/'config.json').write_text(json.dumps({'arch':args.arch,'max_length':64,'threshold':0.5}))
    restored = Classifier(len(vocab), args.arch)
    restored.load_state_dict(torch.load(out/'model.pt', weights_only=True, map_location='cpu'))
    print('教学测试集:', evaluate(restored,loaders['test']))
    ids, lengths, _ = collate([ReviewDataset([{'text':'服务很差，令人失望','label':0}],vocab)[0]])
    restored.eval()
    with torch.inference_mode():
        print('正面分数:', restored(ids,lengths).sigmoid().item())
    print('已保存并重新加载:', str(out))

if __name__ == '__main__': main()
