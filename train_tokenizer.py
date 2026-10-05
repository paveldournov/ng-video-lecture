"""
Train a byte-level BPE tokenizer on a UTF-8 text file, for use by gpt.py and infer.py.

    python train_tokenizer.py
    python train_tokenizer.py --input input.txt --vocab-size 2048 --output tokenizer-eng.json

Byte-level means every possible input can be encoded (there are no unknown characters):
training starts from the 256 byte values and repeatedly merges the most frequent
neighbouring pair into a new token until the vocabulary reaches --vocab-size.
"""
import argparse
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--input', default='warandpeace.txt', help='UTF-8 training text')
parser.add_argument('--vocab-size', type=int, default=4096, help='number of tokens in the vocabulary')
parser.add_argument('--output', default='tokenizer-rus.json', help='where to save the tokenizer')
args = parser.parse_args()

with open(args.input, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = Tokenizer(models.BPE())
# split text into words first (so merges never cross word boundaries), then map each byte to a token
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), # all 256 bytes, even ones absent from the text
    show_progress=False,
)
tokenizer.train_from_iterator([text], trainer=trainer)
tokenizer.save(args.output)

ids = tokenizer.encode(text).ids
assert tokenizer.decode(ids) == text, "round trip failed: decode(encode(text)) != text"
print(f"saved {args.output}: {tokenizer.get_vocab_size()} tokens")
print(f"{len(text):,} characters -> {len(ids):,} tokens ({len(text) / len(ids):.2f} characters per token)")

sample = text[len(text) // 2:][:200].split('\n')[0][:120] # a line from the middle of the text
pieces = [tokenizer.decode([i]) for i in tokenizer.encode(sample).ids]
print(f"example: {sample!r}\n      -> {pieces}")
