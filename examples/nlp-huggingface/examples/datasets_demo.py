"""Load local data, transform, persist, collate variable-length batches."""
import argparse
from pathlib import Path
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

p = argparse.ArgumentParser()
p.add_argument('--model', default='google-bert/bert-base-chinese')
p.add_argument('--revision', default='8f23c25b06e129b6c986331a13d8d025a92cf0ea')
p.add_argument('--out', default='runs/datasets')
args = p.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
raw = load_dataset('csv', data_files=str(Path(__file__).parent/'reviews.csv'))['train']
raw = raw.filter(lambda r: r['text'] is not None and r['text'].strip() != '' and r['label'] in (0,1))
print('单行:', raw[0])
print('切片是列名字典:', raw[:2])
print('一列的前两项:', list(raw['text'][:2]))
splits = DatasetDict({s:raw.filter(lambda r:r['split']==s).remove_columns('split')
                      for s in ('train','validation','test')})
# This extra split demonstrates the API only. It is not used to change our fixed test set.
extra = splits['train'].class_encode_column('label').train_test_split(
    test_size=0.25, seed=42, stratify_by_column='label')
print('训练集内部再次分层抽样:', {k:len(v) for k,v in extra.items()})
tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
def encode(batch):
    return tokenizer(batch['text'], truncation=True, max_length=64)
encoded = splits.map(encode, batched=True, remove_columns=['text'])
encoded = encoded.rename_column('label','labels')
encoded.save_to_disk(str(out/'arrow'))
restored = load_from_disk(str(out/'arrow'))
assert set(restored) == set(encoded)
for split in encoded:
    assert restored[split].to_dict() == encoded[split].to_dict()
    assert restored[split].features == encoded[split].features
splits['train'].to_csv(str(out/'train.csv'), index=False)
splits['train'].to_json(str(out/'train.jsonl'), force_ascii=False)
for fmt, filename in [('csv', 'train.csv'), ('json', 'train.jsonl')]:
    again = load_dataset(fmt, data_files=str(out/filename))['train']
    assert again.to_dict() == splits['train'].to_dict()
    print(fmt, '导出重载的所有记录一致:', len(again))
# with_format returns a new view; set_format changes the existing object's view.
view = restored['train'].with_format('torch')
collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
loader = DataLoader(view, batch_size=4, shuffle=False, collate_fn=collator)
batch = next(iter(loader))
print('动态补齐后的batch:', {k:tuple(v.shape) for k,v in batch.items()})
