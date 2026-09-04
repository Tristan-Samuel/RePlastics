from pathlib import Path


DATA_DIR = Path("data")

TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

EXPECTED_CLASSES = ["metal", "plastic"]
SPLIT_DIRS = {
    "train": TRAIN_DIR,
    "validation": VALIDATION_DIR,
    "test": TEST_DIR,
}


def assert_split_layout():
    for split_name, split_dir in SPLIT_DIRS.items():
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
