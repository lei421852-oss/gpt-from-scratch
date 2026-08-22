import pytest

from gpt.data.tokenizer import CharacterTokenizer


def test_tokenizer_round_trip():
    text = "hello world"
    tokenizer = CharacterTokenizer(text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_tokenizer_save_and_load(tmp_path):
    tokenizer = CharacterTokenizer("cab\n")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    restored = CharacterTokenizer.load(path)

    assert restored.to_dict() == tokenizer.to_dict()
    assert restored.decode(restored.encode("cab\n")) == "cab\n"


def test_tokenizer_rejects_duplicate_vocabulary():
    data = {
        "format_version": 1,
        "type": "character",
        "characters": ["a", "a"],
    }
    with pytest.raises(ValueError, match="duplicate"):
        CharacterTokenizer.from_dict(data)
