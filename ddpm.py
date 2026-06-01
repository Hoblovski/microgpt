"""
ddpm.py: 最小 DDPM（去噪扩散概率模型），仅使用 numpy 做向量化运算。

数据集：UCI optdigits（8x8 灰度手写数字，比 MNIST 更小，~3800 张）
模型  ：MLP 噪声预测器  ε_θ(x_t, t)
        输入 [x_t (64) || time_embed (16)] -> Linear -> SiLU -> Linear -> SiLU -> Linear -> ε_pred (64)
调度  ：linear β, T = 50
训练  ：MSE(eps_pred, eps_true)，手写矩阵反传 + Adam
采样  ：DDPM 反向过程，结果输出为 ASCII art 与 PGM 文件

预期：在普通笔记本 CPU 上 ~1~2 分钟即可收敛，远小于 20 分钟。
"""

import os
import time
import urllib.request
import numpy as np

np.random.seed(42)

# 改这一行：设为路径会自动加载（如存在）/训练后保存；设为 None 则不读不存
weights_file = 'ddpm_ckpt.npz'

# ---------------- 数据集 ----------------
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra'
DATA_PATH = 'optdigits.tra'
if not os.path.exists(DATA_PATH):
    print(f"下载数据集 {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

raw = np.loadtxt(DATA_PATH, delimiter=',', dtype=np.float32)
X = raw[:, :-1]                    # (N, 64)，像素 0..16
X = X / 8.0 - 1.0                  # 归一化到 [-1, 1]
N, D = X.shape
print(f"数据集: {N} 张 8x8 图像, 维度 {D}")

# ---------------- DDPM 噪声调度（cosine, Nichol & Dhariwal 2021） ----------------
T = 100
s = 0.008
_t = np.arange(T + 1, dtype=np.float32) / T
_ab = np.cos((_t + s) / (1 + s) * np.pi / 2) ** 2
_ab = _ab / _ab[0]                                  # 归一化使 α_bar(0)=1
alpha_bars  = _ab[1:]                               # (T,)
alpha_bars_prev = _ab[:-1]
betas       = np.clip(1 - alpha_bars / alpha_bars_prev, 1e-8, 0.999).astype(np.float32)
alphas      = (1.0 - betas).astype(np.float32)
alpha_bars  = alpha_bars.astype(np.float32)
sqrt_ab     = np.sqrt(alpha_bars)
sqrt_1mab   = np.sqrt(1 - alpha_bars)

# 时间嵌入：经典 sinusoidal embedding
T_DIM = 32
def time_embedding(t):
    # t: (B,) int -> (B, T_DIM)
    half = T_DIM // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    args = t[:, None].astype(np.float32) * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=1)

# ---------------- 模型: 4 层 MLP ----------------
H = 256
def init(out, inp):
    # He 初始化（适合 SiLU/ReLU 系激活）
    return np.random.randn(out, inp).astype(np.float32) * np.sqrt(2.0 / inp)

state = {
    'W1': init(H, D + T_DIM), 'b1': np.zeros(H, np.float32),
    'W2': init(H, H),         'b2': np.zeros(H, np.float32),
    'W3': init(H, H),         'b3': np.zeros(H, np.float32),
    'W4': init(D, H),         'b4': np.zeros(D, np.float32),
}
print(f"参数量: {sum(p.size for p in state.values())}")

def sigmoid(x):
    # 数值稳定
    x = np.clip(x, -60.0, 60.0)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def silu(x):
    return x * sigmoid(x)

def silu_back(x):
    s = sigmoid(x)
    return s + x * s * (1.0 - s)

def forward(x_t, t, params=None):
    # x_t: (B, D), t: (B,)
    p = params if params is not None else state
    te = time_embedding(t)                       # (B, T_DIM)
    h0 = np.concatenate([x_t, te], axis=1)       # (B, D+T_DIM)
    z1 = h0 @ p['W1'].T + p['b1']
    a1 = silu(z1)
    z2 = a1 @ p['W2'].T + p['b2']
    a2 = silu(z2)
    z3 = a2 @ p['W3'].T + p['b3']
    a3 = silu(z3)
    z4 = a3 @ p['W4'].T + p['b4']                # (B, D) ε 预测
    cache = (h0, z1, a1, z2, a2, z3, a3)
    return z4, cache

def backward(eps_pred, eps_true, cache):
    h0, z1, a1, z2, a2, z3, a3 = cache
    B = eps_pred.shape[0]
    dz4 = 2.0 * (eps_pred - eps_true) / (B * D)
    g = {}
    g['W4'] = dz4.T @ a3
    g['b4'] = dz4.sum(0)
    da3 = dz4 @ state['W4']
    dz3 = da3 * silu_back(z3)
    g['W3'] = dz3.T @ a2
    g['b3'] = dz3.sum(0)
    da2 = dz3 @ state['W3']
    dz2 = da2 * silu_back(z2)
    g['W2'] = dz2.T @ a1
    g['b2'] = dz2.sum(0)
    da1 = dz2 @ state['W2']
    dz1 = da1 * silu_back(z1)
    g['W1'] = dz1.T @ h0
    g['b1'] = dz1.sum(0)
    return g

# ---------------- Adam ----------------
lr_max  = 2e-3
lr_min  = 1e-4
beta1   = 0.9
beta2   = 0.999
eps_a   = 1e-8
m_buf   = {k: np.zeros_like(v) for k, v in state.items()}
v_buf   = {k: np.zeros_like(v) for k, v in state.items()}

# EMA：权重的指数滑动平均，采样时用它
ema_decay = 0.999
ema = {k: v.copy() for k, v in state.items()}

# ---------------- 训练循环 ----------------
num_epochs = 10000
batch_size = 128
save_every = 200    # 每多少 epoch 自动保存一次（0 表示只在结束时保存）
steps_per_epoch = N // batch_size
total_steps = num_epochs * steps_per_epoch
print(f"总 steps: {total_steps}")

# ---------------- checkpoint 工具 ----------------
def save_ckpt(path, epoch, step):
    blob = {'__meta__': np.array([epoch, step], dtype=np.int64)}
    for k, v in state.items(): blob[f'state/{k}'] = v
    for k, v in ema.items():   blob[f'ema/{k}']   = v
    for k, v in m_buf.items(): blob[f'm/{k}']     = v
    for k, v in v_buf.items(): blob[f'v/{k}']     = v
    # 注意 np.savez 会自动给 fname 追加 .npz，因此这里 tmp 用裸名，replace 时再加 .npz
    tmp_base = path + '.tmp'
    np.savez(tmp_base, **blob)
    os.replace(tmp_base + '.npz', path)        # 原子替换，避免中途崩溃损坏文件

def load_ckpt(path):
    blob = np.load(path)
    for k in state: state[k][...] = blob[f'state/{k}']
    for k in ema:   ema[k][...]   = blob[f'ema/{k}']
    for k in m_buf: m_buf[k][...] = blob[f'm/{k}']
    for k in v_buf: v_buf[k][...] = blob[f'v/{k}']
    e, s = blob['__meta__']
    return int(e), int(s)

start_epoch, step = 0, 0
if weights_file and os.path.exists(weights_file):
    start_epoch, step = load_ckpt(weights_file)
    print(f"已从 {weights_file} 加载 checkpoint (epoch={start_epoch}, step={step})")

t0 = time.time()
for epoch in range(start_epoch, num_epochs):
    perm = np.random.permutation(N)
    epoch_loss = 0.0
    for b in range(steps_per_epoch):
        idx = perm[b * batch_size:(b + 1) * batch_size]
        x0  = X[idx]                                                  # (B, D)
        t   = np.random.randint(0, T, size=batch_size)                # (B,)
        eps = np.random.randn(batch_size, D).astype(np.float32)
        # 前向扩散：x_t = sqrt(α_bar)·x_0 + sqrt(1-α_bar)·ε
        x_t = sqrt_ab[t][:, None] * x0 + sqrt_1mab[t][:, None] * eps

        eps_pred, cache = forward(x_t, t)
        loss = float(np.mean((eps_pred - eps) ** 2))
        epoch_loss += loss

        grads = backward(eps_pred, eps, cache)
        step += 1
        # 余弦学习率衰减
        progress = step / total_steps
        lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))
        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step
        for k in state:
            m_buf[k] = beta1 * m_buf[k] + (1 - beta1) * grads[k]
            v_buf[k] = beta2 * v_buf[k] + (1 - beta2) * grads[k] ** 2
            state[k] -= lr_t * (m_buf[k] / bc1) / (np.sqrt(v_buf[k] / bc2) + eps_a)
            ema[k] = ema_decay * ema[k] + (1 - ema_decay) * state[k]

    if epoch < 3 or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        print(f"epoch {epoch+1:3d}/{num_epochs} | loss {epoch_loss/steps_per_epoch:.4f} "
              f"| 累计 {time.time()-t0:.1f}s")

    if weights_file and save_every and (epoch + 1) % save_every == 0:
        save_ckpt(weights_file, epoch + 1, step)
        print(f"  -> 已保存 checkpoint 到 {weights_file}")

if weights_file and start_epoch < num_epochs:
    save_ckpt(weights_file, num_epochs, step)
    print(f"训练完成，总用时 {time.time()-t0:.1f}s，已保存到 {weights_file}")
else:
    print("跳过训练（已到目标 epoch），直接采样。")

# ---------------- 采样：DDPM 反向过程（用 EMA 权重） ----------------
def sample(n=16):
    x = np.random.randn(n, D).astype(np.float32)   # x_T ~ N(0, I)
    for t_step in range(T - 1, -1, -1):
        t = np.full(n, t_step, dtype=np.int64)
        eps_pred, _ = forward(x, t, params=ema)
        a_t   = alphas[t_step]
        ab_t  = alpha_bars[t_step]
        coef  = (1.0 - a_t) / np.sqrt(1.0 - ab_t)
        mean  = (x - coef * eps_pred) / np.sqrt(a_t)
        if t_step > 0:
            sigma = np.sqrt(betas[t_step])
            x = mean + sigma * np.random.randn(n, D).astype(np.float32)
        else:
            x = mean
    return x

samples = sample(16)
# 反归一化到 0..16
samples = np.clip((samples + 1.0) * 8.0, 0, 16)
samples = samples.reshape(-1, 8, 8)

# ---------------- ASCII 预览 ----------------
chars = ' .:-=+*#%@'
print("\n--- 生成样本 (ASCII) ---")
for i, img in enumerate(samples[:8]):
    print(f"sample {i+1}:")
    for row in img:
        print(''.join(chars[min(int(v / 16 * (len(chars) - 1) + 0.5), len(chars) - 1)] for v in row))
    print()

# ---------------- 保存 4x4 网格 PGM ----------------
grid = np.zeros((4 * 8, 4 * 8), dtype=np.uint8)
for i, img in enumerate(samples):
    r, c = i // 4, i % 4
    grid[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = (img / 16.0 * 255.0).astype(np.uint8)
with open('ddpm_samples.pgm', 'wb') as f:
    f.write(f'P5\n{grid.shape[1]} {grid.shape[0]}\n255\n'.encode())
    f.write(grid.tobytes())
print("已保存 ddpm_samples.pgm")
