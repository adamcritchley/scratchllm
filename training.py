import os
import torch
import torch.nn as nn
from datasets import DatasetDict
from torch.nn import functional as F
import tokenizer
import pickle
import datasets
import argparse
import random

# TODO: use an autotuning script

parser = argparse.ArgumentParser(description='This is a GPT model demo')

parser.add_argument('-batch_size', default=2048, type=str, help='Please provide a batch size')

args = parser.parse_args()

block_size = 2048 # 64
batch_size = args.batch_size # 128
max_iters = 30000
learning_rate = 3e-4 # 3e-3, 3e-4, 1e-3, 1e-4 -> prev 3e-3
eval_iters = 100
n_embd = 650
n_head = 100 # prev 4
n_layer = 200 # prev 4
dropout = 0.2 # 20% dropout neurons
training_split = 0.90

# set n_embd * n_layers * 2 to 1B

tok = tokenizer.Tokenizer(model_path="fun.model")

def load_online_dataset(corpus_url:str):
    return datasets.load_dataset(corpus_url, trust_remote_code=True)

def load_src_dataset(corpus_file:str):
    return open(corpus_file, "r", encoding="utf-8")

def get_src_data(ds, seek):
    if isinstance(ds, DatasetDict):
        return ds['train'][seek]['text']
    else:
        global indices
        lines = ""
        ds.seek(indices[seek])
        for l in ds:
            lines += l
        return lines


def get_rows_online_dataset(ds):
    return ds['train'].num_rows


def get_rows_src_dataset(ds):
    fsize = 0
    indices = []
    for l in ds:
        if l.startswith("int main("):
            indices.append(fsize)
        fsize += len(l)
    return indices

#dataset = load_online_dataset("Skylion007/openwebtext")
#num_dataset_rows = get_rows_online_dataset(dataset)

dataset = load_src_dataset("corpus.txt")
indices = get_rows_src_dataset(dataset)
num_dataset_rows = len(indices)

if num_dataset_rows == 0:
    print("bad dataset!")
    exit(1)

def get_batch(split):
    if split == 'train':
        percent_split = training_split
    else:
        percent_split = 1.0
    di = random.randint(0, int((num_dataset_rows - 1) * percent_split))
    data = torch.tensor(tok.encode(get_src_data(dataset, di), False, False), dtype=torch.long)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        # scaling by the 1/sqrt(length of a row in the keys or queries matrix OR dk) [dk = dimensionality of the key vector]
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out


# %%
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],
                        dim=-1)  # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3]) F = Feature dimension
        out = self.dropout(self.proj(out))
        return out


# %%
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  # clamp neg to zero
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)  # dropout neurons, change them to be zero
        )

    def forward(self, x):
        return self.net(x)


# %%
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, h_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # residual connection, used for add then norm
        self.ln2 = nn.LayerNorm(n_embd)  # residual connection, used for add then norm

    def forward(self, x):
        # post-norm
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


# %%
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeddings_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  # n_head???
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embeddings_table(index)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            # focus only on the last step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1)  # (B, T+1)
        return index


if __name__ == "__main__":
    print("settings")
    print(f"block_size = {block_size}")
    print(f"batch_size = {batch_size}")
    print(f"max_iters = {max_iters}")
    print(f"learning_rate = {learning_rate}")
    print(f"eval_iters = {eval_iters}")
    print(f"n_embd = {n_embd}")
    print(f"n_head = {n_head}")
    print(f"n_layer = {n_layer}")
    print(f"dropout = {dropout}")
    print(f"training_split = {training_split}")
    print(f"total tokens = {tok.n_words}")
    print()
    print("starting model")
    model = GPTLanguageModel(tok.n_words)

    num_params = sum(
       p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"model has {num_params} parameters")

    print("starting optim")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("starting training")
    for iter in range(max_iters):
        if iter % eval_iters == 0:
            losses = estimate_loss()
            print(f"step {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("final loss")
    print(loss.item())

    print("saving model")
    with open('model-01.pkl', 'wb') as f:
        pickle.dump(model, f)

