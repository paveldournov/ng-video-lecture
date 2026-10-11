"""
GPT model definitions shared by gpt.py (training) and infer.py (generation).

GPT     - the current architecture, GPT-2 style: fused causal attention, GELU, input and output
          embeddings tied, scaled init of the residual projections. Checkpoints have config['arch'] == 'gpt2'.
GPTv1   - the original lecture architecture (separate attention heads, ReLU, untied embeddings),
          kept so older checkpoints still load.
generate - sampling with temperature, top-k, top-p and a repetition penalty, for either model.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int # maximum context length in tokens
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """ all heads of masked self-attention in one matrix multiply, using PyTorch's fused attention kernel """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False) # query, key, value for all heads
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble the heads side by side
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd) # position embeddings
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: reading a token in and predicting it out use the same vectors; this saves
        # vocab_size * n_embd parameters and generalizes better on small data
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # every block adds two residual projections to the stream; scale them down so it doesn't grow with depth
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos)) # (B,T,C)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x)) # (B,T,vocab_size)
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, learning_rate, weight_decay, betas):
        # decay only the matrices (linear layers, embeddings); biases and LayerNorm weights are left alone
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {'params': [p for p in params if p.dim() >= 2], 'weight_decay': weight_decay},
            {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=torch.cuda.is_available())


class GPTv1(nn.Module):
    """ the original lecture model; parameter names match the checkpoints it saved """

    class Head(nn.Module):
        def __init__(self, n_embd, head_size, block_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        def forward(self, x):
            B, T, C = x.shape
            k, q = self.key(x), self.query(x)
            wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
            wei = F.softmax(wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')), dim=-1)
            return wei @ self.value(x)

    class MultiHeadAttention(nn.Module):
        def __init__(self, n_embd, n_head, block_size):
            super().__init__()
            self.heads = nn.ModuleList([GPTv1.Head(n_embd, n_embd // n_head, block_size) for _ in range(n_head)])
            self.proj = nn.Linear(n_embd, n_embd)

        def forward(self, x):
            return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

    class FeedFoward(nn.Module):
        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        def __init__(self, n_embd, n_head, block_size):
            super().__init__()
            self.sa = GPTv1.MultiHeadAttention(n_embd, n_head, block_size)
            self.ffwd = GPTv1.FeedFoward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            return x + self.ffwd(self.ln2(x))

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[GPTv1.Block(config.n_embd, config.n_head, config.block_size) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def from_checkpoint(checkpoint):
    """ build the right model class for a checkpoint saved by gpt.py and load its weights """
    config = dict(checkpoint['config'])
    arch = config.pop('arch', 'v1') # checkpoints from before the GPT-2 style model have no 'arch'
    model = (GPT if arch == 'gpt2' else GPTv1)(GPTConfig(**config))
    model.load_state_dict(checkpoint['model'])
    return model


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, repetition_window=64):
    """
    Continue the token sequences in idx (B, T) by max_new_tokens tokens.

    temperature        below 1: safer, more repetitive; above 1: more random
    top_k              sample only from the k most likely tokens
    top_p              sample only from the smallest set of tokens whose probabilities add up to p (nucleus sampling)
    repetition_penalty above 1 makes tokens from the last repetition_window tokens less likely (1.1-1.3 is typical)
    """
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block_size:]) # crop to the last block_size tokens
        logits = logits[:, -1, :].float() # predictions for the last position, (B, vocab_size)
        if repetition_penalty != 1.0:
            recent = idx[:, -repetition_window:]
            scores = logits.gather(1, recent)
            # shrink positive scores and push negative ones further down, so the penalty always lowers the probability
            logits.scatter_(1, recent, torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty))
        logits = logits / temperature
        if top_k is not None:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float('-inf'))
        if top_p is not None:
            sorted_logits, order = torch.sort(logits, descending=True)
            cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # drop a token once the tokens before it already cover top_p (the most likely token always stays)
            drop = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
            logits = logits.masked_fill(torch.zeros_like(drop).scatter(1, order, drop), float('-inf')) # back to vocab order
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
    return idx
