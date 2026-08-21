from pathlib import Path

from torch.utils.data import DataLoader

from gpt.data.tokenizer import CharacterTokenizer
from gpt.data.dataset import LanguageModelDataset
from gpt.data.loader import create_dataloader


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
    block_size=16
)


loader = create_dataloader(
    dataset,
    batch_size=4
)


x, y = next(iter(loader))


print("Input shape:")
print(x.shape)

print("Target shape:")
print(y.shape)


print("\nInput:")
print(x)