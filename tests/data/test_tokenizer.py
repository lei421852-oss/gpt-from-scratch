from gpt.data.tokenizer import CharacterTokenizer


def test_encode_decode():

    text = "hello"

    tokenizer = CharacterTokenizer(text)

    ids = tokenizer.encode("hello")

    recovered = tokenizer.decode(ids)

    assert recovered == "hello"