from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_KAGGLE_ROOTS = (
    Path("/kaggle/input/competitions/recursion-cellular-image-classification"),
    Path("/kaggle/input/recursion-cellular-image-classification"),
)

_LOCAL_IMAGE_ROOT = _PROJECT_ROOT / "data/raw/rxrx1_original_512/images"


def get_kaggle_root():
    for root in _KAGGLE_ROOTS:
        if root.exists():
            return root
    return None


def get_image_root(split="train"):
    kaggle_root = get_kaggle_root()
    root = kaggle_root / split if kaggle_root else _LOCAL_IMAGE_ROOT

    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")
    return root


def get_metadata_path(split="train"):
    kaggle_root = get_kaggle_root()
    if kaggle_root is None:
        raise RuntimeError("Kaggle competition data not found.")

    path = kaggle_root / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    return path