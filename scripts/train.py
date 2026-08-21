import torch

from torch.utils.data import DataLoader

from gpt.data.tokenizer import CharacterTokenizer
from gpt.data.dataset import LanguageModelDataset
from gpt.model.gpt import GPT
from gpt.trainer.trainer import Trainer


# device

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Device:", device)


text = open(
    "data/raw/input.txt",
    encoding="utf-8"
).read()


tokenizer = CharacterTokenizer(text)


tokens = tokenizer.encode(text)


dataset = LanguageModelDataset(
    tokens,
    block_size=128
)


loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


model = GPT(
    vocab_size=tokenizer.vocab_size,
    block_size=128,
    embedding_dim=256,
    num_layers=6,
    num_heads=8
)


model = model.to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


trainer = Trainer(
    model,
    optimizer,
    device
)


for step, (x,y) in enumerate(loader):

    loss = trainer.train_step(
        x,
        y
    )

    if step % 100 == 0:
        print(
            f"step {step}, loss {loss:.4f}"
        )

    if step == 1000:
        break