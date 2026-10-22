"""One-head self-attention with invented vectors; no trained semantics."""
import numpy as np

X = np.array([[1., 0.], [0., 1.], [1., 1.]])
Wq = Wk = Wv = np.eye(2)  # Chosen for arithmetic, not learned weights.
Q, K, V = X @ Wq, X @ Wk, X @ Wv
scores = Q @ K.T / np.sqrt(K.shape[-1])

def softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)

weights = softmax(scores)
output = weights @ V
causal_scores = np.where(np.triu(np.ones((3,3), dtype=bool), k=1), -np.inf, scores)
causal_weights = softmax(causal_scores)
assert np.allclose(weights.sum(axis=-1), 1)
assert np.allclose(np.triu(causal_weights, k=1), 0)
print('普通注意力权重:', weights.round(3), sep='\n')
print('输出:', output.round(3), sep='\n')
print('因果注意力权重:', causal_weights.round(3), sep='\n')
