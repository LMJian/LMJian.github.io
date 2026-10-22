"""Three small translation architectures sharing the same training contract.

The corpus is handwritten and tiny. Test BLEU only demonstrates the evaluator.
It is not a measure of practical translation quality.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from common import seed_everything

TRAIN = [('我喜欢茶','i like tea'), ('你喜欢咖啡','you like coffee'),
         ('我喝水','i drink water'), ('你喝茶','you drink tea'),
         ('他喜欢水','he likes water'), ('她喜欢咖啡','she likes coffee'),
         ('他喝咖啡','he drinks coffee'), ('她喝茶','she drinks tea'),
         ('我喜欢咖啡','i like coffee'), ('你喜欢茶','you like tea'),
         ('他喜欢咖啡','he likes coffee'), ('她喝水','she drinks water')]
VALID = [('我喝茶','i drink tea'), ('他喜欢茶','he likes tea')]
TEST = [('你喝水','you drink water'), ('她喜欢茶','she likes tea')]
PAD, UNK, BOS, EOS = 0, 1, 2, 3

def vocab(tokens):
    return {t:i for i,t in enumerate(['<pad>','<unk>','<bos>','<eos>']+sorted(set(tokens)))}

def batch(pairs, sv, tv):
    src = [torch.tensor([sv.get(c,UNK) for c in s]) for s,_ in pairs]
    tgt = [torch.tensor([BOS]+[tv.get(w,UNK) for w in t.split()]+[EOS]) for _,t in pairs]
    return pad_sequence(src,batch_first=True),pad_sequence(tgt,batch_first=True)

class RecurrentTranslator(nn.Module):
    def __init__(self, ns, nt, attention=False):
        super().__init__()
        self.attention = attention
        self.src_emb, self.tgt_emb = nn.Embedding(ns,24,padding_idx=PAD),nn.Embedding(nt,24,padding_idx=PAD)
        self.encoder = nn.GRU(24,24,batch_first=True,bidirectional=True)
        self.decoder = nn.GRU(24,48,batch_first=True)
        self.head = nn.Linear(96 if attention else 48,nt)

    def encode(self, src):
        lengths = src.ne(PAD).sum(-1)
        packed = pack_padded_sequence(self.src_emb(src),lengths.cpu(),batch_first=True,enforce_sorted=False)
        packed_output,h = self.encoder(packed)
        memory,_ = pad_packed_sequence(packed_output,batch_first=True,total_length=src.shape[1])
        initial = torch.cat([h[-2],h[-1]],-1).unsqueeze(0)
        return memory,initial,src.eq(PAD)

    def decode(self, tgt, encoded):
        memory,initial,src_pad = encoded
        hidden,_ = self.decoder(self.tgt_emb(tgt),initial)
        if self.attention:
            # Dot attention: target queries [B,T,H], source keys/values [B,S,H].
            scores = hidden @ memory.transpose(1,2)
            scores = scores.masked_fill(src_pad.unsqueeze(1),float('-inf'))
            weights = scores.softmax(-1)
            context = weights @ memory
            hidden = torch.cat([hidden,context],-1)
        return self.head(hidden)

    def forward(self,src,tgt): return self.decode(tgt,self.encode(src))

class PositionEncoding(nn.Module):
    def __init__(self,dim,max_len=128):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        rate = torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000)/dim))
        pe = torch.zeros(max_len,dim)
        pe[:,0::2],pe[:,1::2] = torch.sin(position*rate),torch.cos(position*rate)
        self.register_buffer('pe',pe)
    def forward(self,x): return x+self.pe[:x.shape[1]]

class TransformerTranslator(nn.Module):
    def __init__(self,ns,nt):
        super().__init__()
        self.dim = 32
        self.src_emb,self.tgt_emb = nn.Embedding(ns,32,padding_idx=PAD),nn.Embedding(nt,32,padding_idx=PAD)
        self.pos = PositionEncoding(32)
        self.net = nn.Transformer(d_model=32,nhead=4,num_encoder_layers=1,num_decoder_layers=1,
                                  dim_feedforward=64,dropout=0.,batch_first=True)
        # Keep the teaching example on the straightforward dense tensor path.
        self.net.encoder.enable_nested_tensor = False
        self.net.encoder.use_nested_tensor = False
        self.head = nn.Linear(32,nt)

    def encode(self,src):
        src_pad = src.eq(PAD)
        memory = self.net.encoder(self.pos(self.src_emb(src)*math.sqrt(self.dim)),
                                  src_key_padding_mask=src_pad)
        return memory,src_pad

    def decode(self,tgt,encoded):
        memory,src_pad = encoded
        length = tgt.shape[1]
        causal = torch.triu(torch.ones(length,length,dtype=torch.bool,device=tgt.device),diagonal=1)
        hidden = self.net.decoder(self.pos(self.tgt_emb(tgt)*math.sqrt(self.dim)),memory,
                                  tgt_mask=causal,tgt_key_padding_mask=tgt.eq(PAD),
                                  memory_key_padding_mask=src_pad)
        return self.head(hidden)

    def forward(self,src,tgt): return self.decode(tgt,self.encode(src))

@torch.inference_mode()
def generate(model,src,tv,max_new_tokens=12):
    model.eval()
    memory = model.encode(src)
    result = torch.full((1,1),BOS,dtype=torch.long)
    for _ in range(max_new_tokens):
        # Teaching implementation recomputes the prefix; production uses recurrent/KV caches.
        logits = model.decode(result,memory)[:,-1,:].clone()
        logits[:,[PAD,UNK,BOS]] = float('-inf')
        next_id = logits.argmax(-1,keepdim=True)
        result = torch.cat([result,next_id],dim=1)
        if next_id.item() == EOS: break
    id_to_token = {i:t for t,i in tv.items()}
    return [id_to_token[i] for i in result[0].tolist() if i not in (PAD,BOS,EOS)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch',choices=['gru','attention','transformer'],default='attention')
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--out',default='runs/translation')
    args = p.parse_args()
    if args.epochs < 1: p.error('--epochs must be positive')
    seed_everything()
    sv = vocab(c for s,_ in TRAIN for c in s)
    tv = vocab(w for _,t in TRAIN for w in t.split())
    model = (TransformerTranslator(len(sv),len(tv)) if args.arch=='transformer'
             else RecurrentTranslator(len(sv),len(tv),attention=args.arch=='attention'))
    train_src,train_tgt = batch(TRAIN,sv,tv)
    val_src,val_tgt = batch(VALID,sv,tv)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    best,best_state = float('inf'),None
    losses = []
    for _ in range(args.epochs):
        model.train();optimizer.zero_grad()
        logits = model(train_src,train_tgt[:,:-1])
        loss = loss_fn(logits.reshape(-1,len(tv)),train_tgt[:,1:].reshape(-1))
        assert torch.isfinite(loss)
        losses.append(loss.item())
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_logits = model(val_src,val_tgt[:,:-1])
            val_loss = loss_fn(val_logits.reshape(-1,len(tv)),val_tgt[:,1:].reshape(-1)).item()
        if val_loss < best: best,best_state = val_loss,copy.deepcopy(model.state_dict())
    out = Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.save(best_state,out/'model.pt')
    (out/'config.json').write_text(json.dumps({'arch':args.arch,'source_vocab':sv,'target_vocab':tv},ensure_ascii=False),encoding='utf-8')
    model.load_state_dict(torch.load(out/'model.pt',weights_only=True,map_location='cpu'))
    hypotheses,references = [],[]
    for source,target in TEST:
        src,_ = batch([(source,target)],sv,tv)
        words = generate(model,src,tv)
        hypotheses.append(words);references.append([target.split()])
        print(source,'->',' '.join(words),'| reference:',target)
    bleu = corpus_bleu(references,hypotheses,smoothing_function=SmoothingFunction().method1)
    print('训练loss首尾:',round(losses[0],4),round(losses[-1],4))
    print('两条教学测试样本的平滑BLEU-4（0到1）:',round(bleu,4))
    print('样本太少且句子太短，该分数不能用于判断翻译质量或模型优劣。')

if __name__=='__main__': main()
