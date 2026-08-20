from pathlib import Path

from gpt.data.validation import validate_text_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT /
    "data" /
    "raw" /
    "input.txt"
)


if __name__ == "__main__":

    result = validate_text_file(
        DATA_PATH
    )

    print("Dataset validation passed.")

    for key, value in result.items():
        print(f"{key}: {value}")