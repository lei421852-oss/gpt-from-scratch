from gpt.data.dataset import LanguageModelDataset


def test_language_dataset():

    tokens = [
        0,1,2,3,4,5
    ]

    dataset = LanguageModelDataset(
        tokens,
        block_size=3
    )


    x,y = dataset[0]


    assert x.tolist() == [
        0,1,2
    ]

    assert y.tolist() == [
        1,2,3
    ]


def test_dataset_length():

    tokens = [
        0,1,2,3,4,5
    ]

    dataset = LanguageModelDataset(
        tokens,
        block_size=3
    )

    assert len(dataset) == 3