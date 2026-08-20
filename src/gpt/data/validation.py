from pathlib import Path


def validate_text_file(path: str):
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} does not exist"
        )

    if file_path.stat().st_size == 0:
        raise ValueError(
            "Dataset file is empty"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    return {
        "num_characters": len(text),
        "num_lines": text.count("\n"),
        "file_size": file_path.stat().st_size,
    }