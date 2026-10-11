import html
import os
import time
import torch
import math
from tokenizers import Tokenizer
from model import GPT, GPTConfig, generate

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 6e-4 # peak learning rate, reached after warmup_iters
min_lr = 6e-5 # the learning rate decays to this by max_iters (cosine schedule)
warmup_iters = 200
weight_decay = 0.1
betas = (0.9, 0.95)
grad_clip = 1.0 # clip the gradient norm to this, so a bad batch can't throw the weights off
if not torch.cuda.is_available():
    raise SystemExit(f"CUDA GPU not available (torch {torch.__version__}); install a CUDA build of PyTorch.")
else:
    print(f"CUDA GPU available (torch {torch.__version__})")

device = 'cuda'
eval_iters = 200
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.2
input_path = 'data/tolstoy/tolstoy_all.txt' # UTF-8 training text, built by prepare_tolstoy.py
tokenizer_path = 'tokenizer-tolstoy-all.json' # BPE tokenizer trained on input_path by train_tokenizer.py
checkpoint_path = 'gpt-tolstoy-v2.model.pt' # best weights (lowest val loss) are saved here; delete the file to retrain
wandb_project = 'tolstoy-gpt' # Weights & Biases project for live training charts; None turns logging off
log_interval = 20 # send the training batch loss to W&B every this many steps
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# subword tokens instead of characters: run train_tokenizer.py first to create the tokenizer
if not os.path.exists(tokenizer_path):
    raise SystemExit(f"tokenizer {tokenizer_path} not found; run: python train_tokenizer.py --input {input_path} --output {tokenizer_path}")
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()
encode = lambda s: tokenizer.encode(s).ids # encoder: take a string, output a list of integers
decode = lambda l: tokenizer.decode(l) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
# per-token losses aren't comparable across tokenizers, so also report bits per character
chars_per_token = len(text) / len(data)
bits_per_char = lambda loss: loss / chars_per_token / math.log(2)
print(f"{len(text):,} characters -> {len(data):,} tokens, vocab size {vocab_size}")
# validation: every 10th block of 4096 tokens, so it samples all works and genres evenly
# (the last 10% of the file would be mostly letters from the complete-works volumes)
split_blocks = data.split(4096)
train_data = torch.cat([b for i, b in enumerate(split_blocks) if i % 10 != 9])
val_data = torch.cat([b for i, b in enumerate(split_blocks) if i % 10 == 9])

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, dropout=dropout)
model = GPT(config).to(device) # the model itself lives in model.py, shared with infer.py
# print the number of parameters in the model (the tied input/output embedding counts once)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

def get_lr(iter):
    # linear warmup, then cosine decay from learning_rate down to min_lr at max_iters
    if iter < warmup_iters:
        return learning_rate * (iter + 1) / warmup_iters
    progress = (iter - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_lr)

def save_checkpoint(iter, val_loss):
    # everything infer.py needs to rebuild the model without this script or the training text
    torch.save({
        'model': model.state_dict(),
        'config': dict(vars(config), arch='gpt2'), # model.from_checkpoint picks the model class by 'arch'
        'tokenizer': tokenizer_path,
        'iter': iter,
        'val_loss': val_loss,
    }, checkpoint_path)

run = None # W&B run, only while training
if os.path.exists(checkpoint_path):
    # load the trained weights from a previous run instead of training again
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    print(f"loaded model from {checkpoint_path} (step {checkpoint['iter']}, val loss {checkpoint['val_loss']:.4f})")
else:
    # AdamW with weight decay on the weight matrices only; the learning rate is set every step from get_lr
    optimizer = model.configure_optimizer(learning_rate, weight_decay, betas)
    start_time = time.time()
    best_val_loss = float('inf')

    if wandb_project:
        import wandb
        run = wandb.init(project=wandb_project, name=checkpoint_path.removesuffix('.pt').removesuffix('.model'), config=dict(
            batch_size=batch_size, block_size=block_size, max_iters=max_iters, eval_interval=eval_interval,
            eval_iters=eval_iters, learning_rate=learning_rate, min_lr=min_lr, warmup_iters=warmup_iters,
            weight_decay=weight_decay, betas=betas, grad_clip=grad_clip, n_embd=n_embd, n_head=n_head, n_layer=n_layer,
            dropout=dropout, vocab_size=vocab_size, input_path=input_path, tokenizer_path=tokenizer_path,
            train_tokens=len(train_data), val_tokens=len(val_data), chars_per_token=chars_per_token,
            parameters=sum(p.numel() for p in model.parameters()),
        ))
    step_timer, timed_from = time.time(), 0 # wall time and first step of the current timing window

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            elapsed = int(time.time() - start_time)
            # the text is small, so the model eventually memorizes it: train loss keeps falling while val loss
            # turns back up. keep the weights from the step with the best val loss rather than the last step.
            improved = losses['val'] < best_val_loss
            if improved:
                best_val_loss = losses['val'].item()
                save_checkpoint(iter, best_val_loss)
            print(f"[{time.strftime('%H:%M:%S')} +{elapsed // 60}m{elapsed % 60:02d}s] step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                  f" ({bits_per_char(losses['val']):.3f} bits/char){' *saved' if improved else ''}")
            if run:
                run.log({'eval/train_loss': losses['train'].item(), 'eval/val_loss': losses['val'].item(),
                         'eval/val_bits_per_char': bits_per_char(losses['val'].item()), 'eval/best_val_loss': best_val_loss}, step=iter)
            step_timer, timed_from = time.time(), iter # don't count the evaluation in the step time

        lr = get_lr(iter)
        for group in optimizer.param_groups:
            group['lr'] = lr

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        # bfloat16 matrix math: ~1.6x faster on the RTX 3060 with the same loss; weights stay float32
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if run and (iter + 1) % log_interval == 0:
            run.log({'train/batch_loss': loss.item(), 'train/tokens_seen': (iter + 1) * batch_size * block_size,
                     'train/lr': lr, 'train/grad_norm': grad_norm.item(),
                     'perf/sec_per_step': (time.time() - step_timer) / (iter + 1 - timed_from),
                     'perf/gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9}, step=iter)
            step_timer, timed_from = time.time(), iter + 1

    # generate from the best weights, not the (possibly overfit) last ones
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    print(f"saved best model to {checkpoint_path} (step {checkpoint['iter']}, val loss {checkpoint['val_loss']:.4f})")

# generate from the model (eval mode turns off dropout)
model.eval()
context = torch.tensor([encode('\n')], dtype=torch.long, device=device) # start from a newline
sample = decode(generate(model, context, max_new_tokens=500, temperature=0.8, top_p=0.95, repetition_penalty=1.1)[0].tolist())
print(sample)
if run:
    import wandb
    run.summary['best_step'] = checkpoint['iter']
    run.summary['best_val_loss'] = checkpoint['val_loss']
    run.summary['best_val_bits_per_char'] = bits_per_char(checkpoint['val_loss'])
    run.log({'sample': wandb.Html(f"<pre>{html.escape(sample)}</pre>")})
    run.finish()
#open('more.txt', 'w', encoding='utf-8').write(decode(generate(model, context, max_new_tokens=10000)[0].tolist()))
