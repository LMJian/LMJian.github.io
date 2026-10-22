"""Download a real Chinese BERT checkpoint and inspect its tensor contract."""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

p = argparse.ArgumentParser()
p.add_argument('--model', default='google-bert/bert-base-chinese')
p.add_argument('--revision', default='8f23c25b06e129b6c986331a13d8d025a92cf0ea')
p.add_argument('--out', default='runs/bert-local')
args = p.parse_args()
torch.set_num_threads(2)
tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
model = AutoModel.from_pretrained(args.model, revision=args.revision, use_safetensors=True)
tokens = tokenizer.tokenize('我爱自然语言处理')
ids = tokenizer.convert_tokens_to_ids(tokens)
print('tokens:', tokens)
print('ids:', ids)
print('ids转token:', tokenizer.convert_ids_to_tokens(ids))
print('encode:', tokenizer.encode('我爱自然语言处理'))
print('decode:', tokenizer.decode(tokenizer.encode('我爱自然语言处理'), skip_special_tokens=True))
inputs = tokenizer(['物流很快，包装完整', '有破损'], padding=True,
                   truncation=True, max_length=32, return_tensors='pt')
print('inputs:', {k:tuple(v.shape) for k,v in inputs.items()})
pair = tokenizer('订单什么时候发货？','商品今天发出。')
print('句子对token_type_ids:', pair.get('token_type_ids'))
model.eval()
with torch.inference_mode():
    output = model(**inputs)
print('last_hidden_state:', tuple(output.last_hidden_state.shape))
print('pooler_output:', tuple(output.pooler_output.shape))
out = Path(args.out)
model.save_pretrained(out, safe_serialization=True)
tokenizer.save_pretrained(out)
reloaded = AutoModel.from_pretrained(out, local_files_only=True)
reloaded.eval()
with torch.inference_mode():
    assert torch.allclose(output.last_hidden_state, reloaded(**inputs).last_hidden_state, atol=1e-6)
print('本地重载一致:', str(out))
