"""A complete tiny BERT fine-tuning loop; its scores are not a quality benchmark."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, pipeline
from common import seed_everything, reviews, metrics

@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    losses, n, ys, ps = 0., 0, [], []
    for batch in loader:
        output = model(**batch)
        losses += output.loss.item() * len(batch['labels'])
        n += len(batch['labels'])
        ys.extend(batch['labels'].tolist())
        ps.extend(output.logits.argmax(-1).tolist())
    return losses/n, metrics(ys,ps)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='google-bert/bert-base-chinese')
    p.add_argument('--revision', default='8f23c25b06e129b6c986331a13d8d025a92cf0ea')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--freeze-backbone', action='store_true')
    p.add_argument('--out', default='runs/bert-review')
    args = p.parse_args()
    if args.epochs < 1: p.error('--epochs must be positive')
    seed_everything()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    data = DatasetDict({s:Dataset.from_list([{'text':r['text'],'labels':r['label']} for r in rows])
                        for s,rows in reviews().items()})
    data = data.map(lambda b:tokenizer(b['text'],truncation=True,max_length=64),
                    batched=True,remove_columns=['text'])
    collator = DataCollatorWithPadding(tokenizer)
    loaders = {s:DataLoader(ds,batch_size=4,shuffle=s=='train',collate_fn=collator)
               for s,ds in data.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, use_safetensors=True, num_labels=2,
        id2label={0:'negative',1:'positive'}, label2id={'negative':0,'positive':1})
    if args.freeze_backbone:
        for parameter in model.base_model.parameters(): parameter.requires_grad = False
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                                  lr=1e-3 if args.freeze_backbone else 2e-5)
    best = float('inf')
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_backbone: model.base_model.eval()
        for batch in loaders['train']:
            optimizer.zero_grad()
            loss = model(**batch).loss
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
        val_loss, val_metrics = evaluate(model,loaders['validation'])
        print('epoch',epoch+1,'validation',val_loss,val_metrics)
        if val_loss < best:
            best = val_loss
            model.save_pretrained(out,safe_serialization=True)
            tokenizer.save_pretrained(out)
    model = AutoModelForSequenceClassification.from_pretrained(out,local_files_only=True)
    print('教学测试集:', evaluate(model,loaders['test']))
    infer = pipeline('text-classification',model=model,tokenizer=tokenizer,device=-1)
    print('教学模型预测:', infer('包装破损，客服没有回复',truncation=True,max_length=64))
    print('这只验证微调、保存、重载和推理流程，不代表生产质量。')

if __name__ == '__main__': main()
