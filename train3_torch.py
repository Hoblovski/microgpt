"""
train3_torch.py: PyTorch implementation of train3.py.

Same behavior as train3.py:
- Dataset, tokenizer, single-head attention, position embeddings, rmsnorm,
  residual connections, SGD optimizer, and inference

Different from train3.py:
- Uses torch.Tensor autograd instead of the hand-written Value class
- Uses torch.nn.Parameter / torch.optim.SGD for parameters and updates
"""

import argparse
import os
import random
import urllib.request

import torch
import torch.nn.functional as F


random.seed(42)
torch.manual_seed(42)


def load_docs(path="input.txt"):
    if not os.path.exists(path):
        names_url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
        urllib.request.urlretrieve(names_url, path)
    with open(path, encoding="utf-8") as f:
        docs = [line.strip() for line in f.read().strip().split("\n") if line.strip()]
    random.shuffle(docs)
    return docs


class SingleHeadGPT(torch.nn.Module):
    def __init__(self, vocab_size, n_embd=16, block_size=16):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        self.params = torch.nn.ParameterDict(
            {
                "wte": self._matrix(vocab_size, n_embd),
                "wpe": self._matrix(block_size, n_embd),
                "attn_wq": self._matrix(n_embd, n_embd),
                "attn_wk": self._matrix(n_embd, n_embd),
                "attn_wv": self._matrix(n_embd, n_embd),
                "attn_wo": self._matrix(n_embd, n_embd),
                "mlp_fc1": self._matrix(4 * n_embd, n_embd),
                "mlp_fc2": self._matrix(n_embd, 4 * n_embd),
                "lm_head": self._matrix(vocab_size, n_embd),
            }
        )

    @staticmethod
    def _matrix(nout, nin, std=0.08):
        return torch.nn.Parameter(torch.randn(nout, nin) * std)

    @staticmethod
    def rmsnorm(x):
        ms = (x * x).mean()
        scale = torch.rsqrt(ms + 1e-5)
        return x * scale

    def linear(self, x, name):
        return F.linear(x, self.params[name])

    def gpt(self, token_id, pos_id, keys, values):
        tok_emb = self.params["wte"][token_id]
        pos_emb = self.params["wpe"][pos_id]
        x = tok_emb + pos_emb
        x = self.rmsnorm(x)

        # 1) Single-head attention block
        x_residual = x
        x = self.rmsnorm(x)
        q = self.linear(x, "attn_wq")
        k = self.linear(x, "attn_wk")
        v = self.linear(x, "attn_wv")
        keys.append(k)
        values.append(v)

        key_stack = torch.stack(keys)
        value_stack = torch.stack(values)
        attn_logits = key_stack @ q / (self.n_embd**0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)
        x_attn = attn_weights @ value_stack

        x = self.linear(x_attn, "attn_wo")
        x = x + x_residual

        # 2) MLP block
        x_residual = x
        x = self.rmsnorm(x)
        x = self.linear(x, "mlp_fc1")
        x = F.relu(x)
        x = self.linear(x, "mlp_fc2")
        x = x + x_residual

        logits = self.linear(x, "lm_head")
        return logits


def train(model, docs, stoi, bos, *, num_steps, learning_rate):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [bos] + [stoi[ch] for ch in doc] + [bos]
        n = min(model.block_size, len(tokens) - 1)

        keys, values = [], []
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = model.gpt(token_id, pos_id, keys, values)
            target = torch.tensor([target_id], dtype=torch.long, device=logits.device)
            losses.append(F.cross_entropy(logits.view(1, -1), target))
        loss = torch.stack(losses).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        lr_t = learning_rate * (1 - step / num_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr_t
        optimizer.step()

        if step < 5 or step % 200 == 0:
            print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}")


@torch.no_grad()
def sample(model, uchars, bos, *, num_samples=20, temperature=0.5):
    vocab_size = len(uchars) + 1
    print("\n--- inference (new, hallucinated names) ---")
    for sample_idx in range(num_samples):
        keys, values = [], []
        token_id = bos
        out = []
        for pos_id in range(model.block_size):
            logits = model.gpt(token_id, pos_id, keys, values)
            probs = F.softmax(logits / temperature, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()
            if token_id == bos:
                break
            out.append(uchars[token_id])
        print(f"sample {sample_idx+1:2d}: {''.join(out)}")


def main():
    parser = argparse.ArgumentParser(description="Train the train3.py model with PyTorch autograd.")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    docs = load_docs()
    print(f"num docs: {len(docs)}")

    uchars = sorted(set("".join(docs)))
    stoi = {ch: i for i, ch in enumerate(uchars)}
    bos = len(uchars)
    vocab_size = len(uchars) + 1
    print(f"vocab size: {vocab_size}")

    model = SingleHeadGPT(vocab_size)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num params: {num_params}")

    train(model, docs, stoi, bos, num_steps=args.num_steps, learning_rate=args.learning_rate)
    sample(model, uchars, bos, temperature=args.temperature)


if __name__ == "__main__":
    main()
