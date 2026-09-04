from pathlib import Path
import os


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_KAGGLE_ROOTS = (
    Path("/kaggle/input/competitions/recursion-cellular-image-classification"),
    Path("/kaggle/input/recursion-cellular-image-classification"),
)

_KAGGLE_DATASETS_ROOT = Path("/kaggle/input/datasets")

_LOCAL_IMAGE_ROOT = (
    _PROJECT_ROOT
    / "data/raw/rxrx1_original_512/images"
)


def get_data_root():
    env_root = os.getenv("RXRX1_DATA_ROOT")

    if env_root:
        root = Path(env_root)

        if not root.exists():
            raise FileNotFoundError(
                f"RXRX1_DATA_ROOT does not exist: {root}"
            )

        return root

    for root in _KAGGLE_ROOTS:
        if root.exists():
            return root

    if _KAGGLE_DATASETS_ROOT.exists():
        for root in _KAGGLE_DATASETS_ROOT.glob(
            "*/recursion-cellular-image-classification"
        ):
            if root.exists():
                return root

    return None


def get_image_root(split="train"):
    data_root = get_data_root()

    if data_root is not None:
        root = data_root / split
    else:
        root = _LOCAL_IMAGE_ROOT

    if not root.exists():
        raise FileNotFoundError(
            f"Image root not found: {root}"
        )

    return root


def get_metadata_path(split="train"):
    data_root = get_data_root()

    if data_root is None:
        raise RuntimeError(
            "RxRx1 data root not found."
        )

    path = data_root / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {path}"
        )

    return path