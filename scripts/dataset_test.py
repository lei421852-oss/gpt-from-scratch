from pathlib import Path

from gpt.data.tokenizer import CharacterTokenizer
from gpt.data.dataset import LanguageModelDataset


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


tokens = tokenizer.encode(text)


dataset = LanguageModelDataset(
    tokens,
    block_size=8
)


x,y = dataset[0]


print("Input:")
print(x)

print("Target:")
print(y)


print()

print(
    "Decoded input:"
)

print(
    tokenizer.decode(
        x.tolist()
    )
)