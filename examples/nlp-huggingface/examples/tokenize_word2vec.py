"""Tokenization and vector save/load; this corpus cannot teach reliable semantics."""
import argparse
from pathlib import Path
import jieba
import torch
from gensim.models import Word2Vec, KeyedVectors
from common import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='runs/word2vec')
args = parser.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
seed_everything()
text = '小林毕业于北京大学计算机系'
print('精确模式:', jieba.lcut(text))
print('全模式:', jieba.lcut(text, cut_all=True))
print('搜索模式:', jieba.lcut_for_search(text))
(out/'userdict.txt').write_text('订单履约 100000 n\n', encoding='utf-8')
jieba.load_userdict(str(out/'userdict.txt'))
jieba.add_word('售后工单')
print('业务词典:', jieba.lcut('订单履约异常需要售后工单'))
jieba.del_word('售后工单')

sentences = [['物流','很快'], ['配送','很快'], ['物流','及时'], ['配送','及时'],
             ['质量','很好'], ['服务','很好'], ['质量','很差'], ['服务','很差']]
model = Word2Vec(sentences, vector_size=16, window=2, min_count=1,
                 sg=1, sample=0, workers=1, seed=42, epochs=30)
model.save(str(out/'word2vec.model'))
model = Word2Vec.load(str(out/'word2vec.model'))
model.wv.save_word2vec_format(str(out/'vectors.txt'), binary=False)
vectors = KeyedVectors.load_word2vec_format(str(out/'vectors.txt'), binary=False)
print('维度:', vectors.vector_size)
print('物流向量形状:', vectors['物流'].shape)
print('余弦相似度（仅演示API）:', float(vectors.similarity('物流','配送')))
print('近邻（不代表可靠语义）:', vectors.most_similar('物流', topn=2))

# Align by our own vocabulary order; vector-file row order is not token ID order.
vocab = ['<pad>', '<unk>', '物流', '配送', '售后']
matrix = torch.randn(len(vocab), vectors.vector_size) * 0.02
matrix[0] = 0
for i, token in enumerate(vocab):
    if token in vectors: matrix[i] = torch.from_numpy(vectors[token].copy())
embedding = torch.nn.Embedding.from_pretrained(matrix, freeze=False, padding_idx=0)
print('Embedding输出形状:', tuple(embedding(torch.tensor([[2,3,0]])).shape))
