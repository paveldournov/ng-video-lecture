"""
Generate text from a GPT checkpoint saved by gpt.py, without training.

    python infer.py
    python infer.py --prompt "Князь Андрей" --tokens 300 --seed 42
    python infer.py --temperature 0.8 --top-p 0.95 --repetition-penalty 1.1

The checkpoint stores the model size, architecture and tokenizer file name, so nothing here has to be
kept in sync with gpt.py. Both the current (GPT-2 style) and the older lecture-style checkpoints load.
"""
import argparse
import os
import torch
from tokenizers import Tokenizer
from model import from_checkpoint, generate

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--checkpoint', default='gpt-tolstoy-v2.model.pt', help='weights saved by gpt.py')
parser.add_argument('--prompt', default='\n', help='text to start generating from (default: a newline)')
parser.add_argument('--tokens', type=int, default=200, help='number of tokens to generate (~3.5 characters each)')
parser.add_argument('--temperature', type=float, default=1.0, help='below 1: safer, more repetitive; above 1: more random')
parser.add_argument('--top-k', type=int, default=None, help='sample only from the k most likely tokens')
parser.add_argument('--top-p', type=float, default=None, help='sample only from the most likely tokens covering this much probability, e.g. 0.95')
parser.add_argument('--repetition-penalty', type=float, default=1.0, help='above 1 discourages repeating recent tokens, e.g. 1.1')
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

tokenizer = Tokenizer.from_file(checkpoint['tokenizer'])
encode = lambda s: tokenizer.encode(s).ids # encoder: take a string, output a list of integers
decode = lambda l: tokenizer.decode(l) # decoder: take a list of integers, output a string

model = from_checkpoint(checkpoint).to(device)
model.eval()

context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
out = generate(model, context, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k,
               top_p=args.top_p, repetition_penalty=args.repetition_penalty)
print(decode(out[0].tolist()))
