from pathlib import Path
from urllib.request import urlretrieve


DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/"
    "char-rnn/master/data/tinyshakespeare/input.txt"
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "input.txt"


def download_dataset():
    RAW_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if RAW_DATA_PATH.exists():
        print("Dataset already exists.")
        return

    print("Downloading Tiny Shakespeare dataset...")

    urlretrieve(
        DATA_URL,
        RAW_DATA_PATH
    )

    print(
        f"Dataset saved to {RAW_DATA_PATH}"
    )


if __name__ == "__main__":
    download_dataset()