import json
from pathlib import Path


class CharacterTokenizer:
    FORMAT_VERSION = 1

    def __init__(self, text: str):
        self._initialize(sorted(set(text)))

    def _initialize(self, characters):
        if not characters:
            raise ValueError("Tokenizer vocabulary cannot be empty")
        if len(set(characters)) != len(characters):
            raise ValueError("Tokenizer vocabulary contains duplicate characters")
        if any(not isinstance(character, str) or len(character) != 1 for character in characters):
            raise ValueError("Every tokenizer entry must be one character")

        self.vocab_size = len(characters)
        self.char_to_id = {
            character: index
            for index, character in enumerate(characters)
        }
        self.id_to_char = {
            index: character
            for character, index in self.char_to_id.items()
        }

    def encode(self, text: str):
        return [self.char_to_id[character] for character in text]

    def decode(self, ids):
        return "".join(self.id_to_char[token_id] for token_id in ids)

    def to_dict(self):
        return {
            "format_version": self.FORMAT_VERSION,
            "type": "character",
            "characters": [
                self.id_to_char[index]
                for index in range(self.vocab_size)
            ],
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported tokenizer format version")
        if data.get("type") != "character":
            raise ValueError("Unsupported tokenizer type")
        if not isinstance(data.get("characters"), list):
            raise TypeError("Tokenizer characters must be a list")

        tokenizer = cls.__new__(cls)
        tokenizer._initialize(data["characters"])
        return tokenizer

    def save(self, path):
        tokenizer_path = Path(path)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        tokenizer_path = Path(path)
        data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
