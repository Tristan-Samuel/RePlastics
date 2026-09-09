from pathlib import Path


DATA_DIR = Path("data")

TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

EXPECTED_CLASSES = ["metal", "plastic"]


def split_dirs_for(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    return {
        "train": data_dir / "train",
        "validation": data_dir / "validation",
        "test": data_dir / "test",
    }


SPLIT_DIRS = split_dirs_for(DATA_DIR)


def ensure_split_layout(data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    for split_name, split_dir in split_dirs_for(data_dir).items():
        for class_name in EXPECTED_CLASSES:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)


def assert_split_layout(data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    for split_name, split_dir in split_dirs_for(data_dir).items():
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {split_dir}. "
                "Run python split_data.py first, or create "
                f"{split_name}/metal and {split_name}/plastic "
                "yourself."
            )

        for class_name in EXPECTED_CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing {class_dir}. "
                    "Create it so you can add images there."
                )
