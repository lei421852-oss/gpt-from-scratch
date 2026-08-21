from pathlib import Path

from gpt.data.tokenizer import CharacterTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT /
    "data" /
    "raw" /
    "input.txt"
)


with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as f:
    text = f.read()


tokenizer = CharacterTokenizer(text)


print(
    "Vocabulary size:",
    tokenizer.vocab_size
)


sample = text[:100]

ids = tokenizer.encode(sample)

print("Original:")
print(sample)

print("Encoded:")
print(ids[:20])

print("Decoded:")
print(tokenizer.decode(ids))