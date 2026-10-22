"""A cheap character n-gram baseline; evaluation uses the held-out toy test split."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from common import reviews, metrics

data = reviews()
model = make_pipeline(TfidfVectorizer(analyzer='char', ngram_range=(1,2)),
                      LogisticRegression(max_iter=300, random_state=42))
model.fit([r['text'] for r in data['train']], [r['label'] for r in data['train']])
pred = model.predict([r['text'] for r in data['test']])
print('教学测试集:', metrics([r['label'] for r in data['test']], pred))
print('预测:', model.predict(['物流太慢，服务很差']).tolist())
