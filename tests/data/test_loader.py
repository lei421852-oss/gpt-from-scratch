from gpt.data.dataset import LanguageModelDataset
from gpt.data.loader import create_dataloader


def test_dataloader_shape():

    tokens = list(range(20))

    dataset = LanguageModelDataset(
        tokens,
        block_size=5
    )


    loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False
    )


    x, y = next(iter(loader))


    assert x.shape == (4,5)
    assert y.shape == (4,5)