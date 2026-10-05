"""
Generate text from a GPT checkpoint saved by gpt.py, without training.

    python infer.py
    python infer.py --prompt "Князь Андрей" --tokens 300 --seed 42
    python infer.py --temperature 0.8 --top-k 50

The checkpoint stores the model size and the tokenizer file name, so nothing here
has to be kept in sync with gpt.py. The model definition mirrors gpt.py (minus
dropout and the training loss).
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--checkpoint', default='gpt-rus-bpe.model.pt', help='weights saved by gpt.py')
parser.add_argument('--prompt', default='\n', help='text to start generating from (default: a newline)')
parser.add_argument('--tokens', type=int, default=200, help='number of tokens to generate (~3.5 characters each)')
parser.add_argument('--temperature', type=float, default=1.0, help='below 1: safer, more repetitive; above 1: more random')
parser.add_argument('--top-k', type=int, default=None, help='sample only from the k most likely tokens')
parser.add_argument('--seed', type=int, default=None, help='random seed for reproducible output')
args = parser.parse_args()

if not os.path.exists(args.checkpoint):
    raise SystemExit(f"checkpoint {args.checkpoint} not found; run gpt.py first to train and save the model.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.seed is not None:
    torch.manual_seed(args.seed)

checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
if 'config' not in checkpoint:
    raise SystemExit(f"{args.checkpoint} is a character-level checkpoint from the old gpt.py; "
                     f"use the old infer.py: git show 5b8fec6:infer.py > infer_char.py")

# hyperparameters, read from the checkpoint
config = checkpoint['config']
vocab_size = config['vocab_size']
block_size = config['block_size']
n_embd = config['n_embd']
n_head = config['n_head']
n_layer = config['n_layer']

tokenizer = Tokenizer.from_file(checkpoint['tokenizer'])
encode = lambda s: tokenizer.encode(s).ids # encoder: take a string, output a list of integers
decode = lambda l: tokenizer.decode(l) # decoder: take a list of integers, output a string

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        v = self.value(x) # (B,T,hs)
        return wei @ v # (B, T, hs)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = self.blocks(tok_emb + pos_emb) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        return self.lm_head(x) # (B,T,vocab_size)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop to the last block_size tokens
            logits = self(idx_cond)[:, -1, :] / temperature # predictions for the last time step, (B, C)
            if top_k is not None:
                # drop every token less likely than the k-th most likely one
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float('-inf'))
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel().to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k)
print(decode(out[0].tolist()))
