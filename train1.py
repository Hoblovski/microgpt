"""
train1.py: Bigram language model with a single-layer MLP, trained by gradient descent.

Same as train0.py:
- Dataset, tokenizer, training loop structure, inference

Different from train0.py:
- Model is a neural network (MLP) instead of a count table
- Training is gradient descent (SGD) instead of counting
- Introduces: softmax, linear, numerical and analytic gradients

The MLP is effectively a differentiable version of the bigram count table:
token_id -> embedding lookup -> hidden layer -> logits -> softmax -> probs.
The gradient tells us how to nudge each parameter to reduce the loss. We show
two ways to compute it: numerically (perturb and measure) and analytically
(chain rule). They agree, but the analytic version is O(params) faster.
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
from utils import shape
random.seed(42)

# Dataset: load and tokenize a list of names
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()] # list[str] of documents
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Tokenizer: character-level, with a special BOS (Beginning of Sequence) token
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Initialize the parameters
n_embd = 16     # embedding dimension
matrix = lambda nout, nin: [[random.gauss(0, 0.08) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    'wte': matrix(vocab_size, n_embd), # word token embeddings
    'mlp_fc1': matrix(4 * n_embd, n_embd),
    'mlp_fc2': matrix(vocab_size, 4 * n_embd),
}
params = [(row, j) for mat in state_dict.values() for row in mat for j in range(len(row))]
# 我们需要一个方法来修改参数，直接存参数的值不行，而 Python 又没有指针，就只好用 (parent_list, index) 来表示。
print(f"num params: {len(params)}")

# Model: token_id -> logits
def linear(x, w): # 其实是 Wx，其中 x 是向量，w 是矩阵
    res = [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
    print(f'linear, x={shape(x)}, w={shape(w)}, res={shape(res)}')
    return res

def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    # softmax 有平移不变性质：假设 C 是常数，x 是向量，那么
    #       softmax(x) == softmax(x + C)
    # 所以这里，C 是 -maxval，让 logits 平移后位于 (-oo, 0]，exp 计算出来是 [1, e^1]
    # 
    # 一个前提是，exp(x) 的 x 一定不能太大，否则计算出来就是 +inf 了
    # 关键词：溢出
    total = sum(exps)
    return [e / total for e in exps]

def mlp(token_id):
    # token id: scalar
    # 向量表示都是列优先的，没有 batching i.e. 可以认为 [dim] 向量是 [dim, 1]
    # scalar --> [n_embd] -> [4 * n_embed] -> [vocab_size]
    x = state_dict['wte'][token_id]
    x = linear(x, state_dict['mlp_fc1'])
    x = [max(0, xi) for xi in x] # relu
    logits = linear(x, state_dict['mlp_fc2'])
    return logits

# Forward pass: run the model on a token sequence, return the average loss
def forward(tokens, n):
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = mlp(token_id)
        probs = softmax(logits)
        loss_t = -math.log(probs[target_id])
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)
    return loss

# Two ways to compute the gradient of the loss w.r.t. all parameters:

def numerical_gradient(tokens, n):
    """Perturb each parameter by eps, measure change in loss."""
    loss = forward(tokens, n)
    eps = 1e-5
    grad = []
    for mat in state_dict.values():
        for row in mat:
            for j in range(len(row)):
                old = row[j]
                row[j] = old + eps
                loss_plus = forward(tokens, n)
                row[j] = old
                grad.append((loss_plus - loss) / eps)
    return loss, grad

def analytic_gradient(tokens, n):
    """Derive the gradient analytically using the chain rule."""
    grad = {k: [[0.0] * len(row) for row in mat] for k, mat in state_dict.items()}
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        # forward pass (saving intermediates for backward)
        x = list(state_dict['wte'][token_id])
        h_pre = linear(x, state_dict['mlp_fc1'])
        h = [max(0, v) for v in h_pre]  # relu
        logits = linear(h, state_dict['mlp_fc2'])
        probs = softmax(logits)
        loss_t = -math.log(probs[target_id])
        losses.append(loss_t)
        # backward pass: chain rule, layer by layer
        # d(loss)/d(logits) for softmax + cross-entropy = probs - one_hot(target)
        dlogits = [p / n for p in probs]
        dlogits[target_id] -= 1.0 / n
        # d(loss)/d(mlp_fc2), d(loss)/d(h): logits = mlp_fc2 @ h
        dh = [0.0] * len(h)
        for i in range(len(dlogits)):
            for j in range(len(h)):
                grad['mlp_fc2'][i][j] += dlogits[i] * h[j]
                dh[j] += state_dict['mlp_fc2'][i][j] * dlogits[i]
        # d(loss)/d(h_pre): relu backward
        dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(len(h_pre))]
        # d(loss)/d(mlp_fc1), d(loss)/d(x): h_pre = mlp_fc1 @ x
        dx = [0.0] * len(x)
        for i in range(len(dh_pre)):
            for j in range(len(x)):
                grad['mlp_fc1'][i][j] += dh_pre[i] * x[j]
                dx[j] += state_dict['mlp_fc1'][i][j] * dh_pre[i]
        # d(loss)/d(wte[token_id]): x = wte[token_id]
        for j in range(len(x)):
            grad['wte'][token_id][j] += dx[j]
    loss = (1 / n) * sum(losses)
    grad_flat = [g for mat in grad.values() for row in mat for g in row]
    return loss, grad_flat

# Gradient check: verify numerical and analytic gradients agree
doc = docs[0]
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
n = len(tokens) - 1
loss_n, grad_n = numerical_gradient(tokens, n)
loss_a, grad_a = analytic_gradient(tokens, n)
grad_diff = max(abs(gn - ga) for gn, ga in zip(grad_n, grad_a))
print(f"gradient check | loss_n {loss_n:.6f} | loss_a {loss_a:.6f} | max diff {grad_diff:.8f}")

# Train the model
num_steps = 1000
learning_rate = 1.0
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = len(tokens) - 1

    # Forward + backward pass
    loss, grad = analytic_gradient(tokens, n)

    # SGD update
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, (row, j) in enumerate(params):
        row[j] -= lr_t * grad[i]

    if step < 5 or step % 200 == 0: # print a bit less often
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss:.4f}")

# Inference: sample new names from the model
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    token_id = BOS
    sample = []
    for pos_id in range(16):
        logits = mlp(token_id)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=probs)[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
