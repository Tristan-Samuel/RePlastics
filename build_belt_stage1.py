#!/usr/bin/env python3
"""Build a 224x224 metal/plastic tree for Colab and zip it.

Old photos are stage1_binary_v2 (already 224). Newer photos are the belt dump in data_sep16 plus the Sep 28-30 cans.
Sep 28-30 cans keep a contiguous holdout; the rest of those days go to train.
The two-frame gap on each side of a holdout goes to test. Extra belt and
stage1 plastic test photos move into train.

Stage1 synthetic NonPlastic copies are dropped. Real stage1 NonPlastic
training photos are thinned so they match the belt-can count in train.
Stage1 metal validation and the original test sample stay at 40. Other
real stage1 NonPlastic photos start in test, then move into validation
until metal validation and metal test differ by at most one. Locked
test stems stay in test. A 5-second burst moves together. The six foil
trays the current checkpoint calls plastic are pinned three to each side.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import hashlib
import os
import random
import shutil
import zipfile

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
STAGE1 = Path.home() / "Downloads" / "stage1_binary_v2_256"
BELT = ROOT / "data_sep16"
LOCK_FILE = ROOT / "stage1_locked_test_stems.txt"
DEST = ROOT / "data_belt_stage1"
ZIP_PATH = Path.home() / "Downloads" / "belt_stage1_224.zip"

NEW_CAN_DIRS = (
    ("sep28", Path.home() / "Downloads" / "2026-09-28" / "metal cans"),
    ("sep29", Path.home() / "Downloads" / "2026-09-29" / "metal_cans"),
    ("sep30", Path.home() / "Downloads" / "2026-09-30" / "metal cans"),
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RANDOM_SEED = 42
NEW_CAN_VAL = 40
NEW_CAN_TEST = 84
EDGE_GAP = 2
OLD_METAL_VAL_KEEP = 40
OLD_METAL_TEST_KEEP = 40
OLD_PLASTIC_TEST_KEEP = 60
BELT_PLASTIC_TEST_KEEP = 90
LONG_SIDE = 224
JPEG_QUALITY = 90
BURST_GAP_SECONDS = 5
# Foil trays the belt stage 1 checkpoint calls plastic
# (analysis/belt-stage1/stage1_excluded/scores.tsv). 00501 and 00537
# are locked test stems, so that pair stays in test. 00504 is the same
# pass as 00501.
SHEET_FAIL_TO_VAL = frozenset(
    {
        "Others_capture_00307_20251205_183241_392897",
        "Others_capture_00490_20251205_185518_884539",
        "Others_capture_00660_20251205_190834_342101",
    }
)
SHEET_FAIL_STAY_TEST = frozenset(
    {
        "Others_capture_00501_20251205_185553_949302",
        "Others_capture_00504_20251205_185602_814118",
        "Others_capture_00537_20251205_185804_852017",
    }
)


def image_files(folder):
    return sorted(
        path
        for path in Path(folder).iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.name != ".DS_Store"
    )


def load_locked_test():
    locked = {"metal": set(), "plastic": set()}
    for line in LOCK_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        class_name, stem = line.split("\t", 1)
        locked[class_name].add(stem)
    return locked


def allocate(sizes, total):
    raw = [total * size / sum(sizes) for size in sizes]
    counts = [int(value) for value in raw]
    leftover = total - sum(counts)
    order = sorted(
        range(len(sizes)),
        key=lambda index: (raw[index] - counts[index], sizes[index]),
        reverse=True,
    )
    for index in order[:leftover]:
        counts[index] += 1
    return counts


def timestamp(path):
    for part in path.stem.split("_"):
        if len(part) == 8 and part.isdigit() and part.startswith("2026"):
            date = part
            continue
        if len(part) == 6 and part.isdigit() and "date" in locals():
            return datetime.strptime(date + part, "%Y%m%d%H%M%S")
    raise ValueError(f"No timestamp in {path.name}")


def place_windows(count, n_test, n_val, gap):
    test_start = gap + count // 5
    test_end = test_start + n_test
    val_start = test_end + gap + count // 5
    val_end = val_start + n_val
    if val_end + gap > count:
        val_end = count - gap
        val_start = val_end - n_val
    if val_start < test_end + gap:
        raise SystemExit(
            f"Holdout windows overlap in a day of {count} images "
            f"(test {n_test}, val {n_val})."
        )

    test_index = set(range(test_start, test_end))
    val_index = set(range(val_start, val_end))
    excluded = set()
    for start, end in ((test_start, test_end), (val_start, val_end)):
        excluded.update(range(start - gap, start))
        excluded.update(range(end, end + gap))
    excluded = {index for index in excluded if 0 <= index < count}
    if excluded & test_index or excluded & val_index:
        raise SystemExit("Gap frames landed inside a holdout window.")
    return test_index, val_index, excluded


def assign_new_cans():
    days = []
    for origin, folder in NEW_CAN_DIRS:
        files = image_files(folder)
        files.sort(key=timestamp)
        days.append((origin, files))

    val_counts = allocate(
        [len(files) for _, files in days],
        NEW_CAN_VAL,
    )
    test_counts = allocate(
        [len(files) for _, files in days],
        NEW_CAN_TEST,
    )

    assigned = []
    for (origin, files), n_val, n_test in zip(days, val_counts, test_counts):
        test_index, val_index, excluded = place_windows(
            len(files),
            n_test,
            n_val,
            EDGE_GAP,
        )
        for index, path in enumerate(files):
            if index in test_index:
                split = "test"
            elif index in val_index:
                split = "validation"
            elif index in excluded:
                split = "test"
            else:
                split = "train"
            assigned.append((split, origin, path))
        print(
            f"  {origin}: {len(files)} cans, "
            f"val {n_val} at the later window, "
            f"test {n_test} at the earlier window, "
            f"gap sent to test {len(excluded)}",
            flush=True,
        )
    return assigned


def sample_keep(paths, keep, rng):
    paths = sorted(paths, key=lambda path: path.name)
    if keep > len(paths):
        raise SystemExit(f"Need {keep} photos, only have {len(paths)}.")
    chosen = rng.sample(paths, keep)
    chosen_ids = {path for path in chosen}
    return chosen, [path for path in paths if path not in chosen_ids]


def stage1_assignments(can_train_count):
    locked = load_locked_test()
    rng = random.Random(RANDOM_SEED)
    dropped = Counter()

    train_metal = []
    test_metal = []
    for path in image_files(STAGE1 / "train" / "NonPlastic"):
        if "synth" in path.stem.lower():
            dropped["synth_nonplastic"] += 1
            continue
        if path.stem in locked["metal"]:
            test_metal.append(path)
        else:
            train_metal.append(path)

    val_metal = image_files(STAGE1 / "val" / "NonPlastic")
    for path in val_metal:
        if "synth" in path.stem.lower():
            raise SystemExit(f"Unexpected synthetic val photo: {path.name}")

    kept_train, removed_train = sample_keep(
        train_metal,
        can_train_count,
        rng,
    )
    kept_val, removed_val = sample_keep(
        val_metal,
        OLD_METAL_VAL_KEEP,
        rng,
    )
    kept_test, removed_test = sample_keep(
        test_metal,
        OLD_METAL_TEST_KEEP,
        rng,
    )
    dropped["to_test_from_train"] = len(removed_train)
    dropped["to_test_from_val"] = len(removed_val)
    dropped["to_test_from_locked"] = len(removed_test)

    assigned = []
    for split, paths in (
        ("train", kept_train),
        ("validation", kept_val),
        ("test", kept_test + removed_train + removed_val + removed_test),
    ):
        for path in paths:
            assigned.append((split, "metal", "stage1", path, False))

    train_plastic = image_files(STAGE1 / "train" / "Plastic")
    val_plastic = image_files(STAGE1 / "val" / "Plastic")
    plastic_test = [
        path for path in train_plastic if path.stem in locked["plastic"]
    ]
    kept_plastic_test, moved_plastic_test = sample_keep(
        plastic_test,
        OLD_PLASTIC_TEST_KEEP,
        rng,
    )
    dropped["plastic_test_moved_to_train"] = len(moved_plastic_test)
    moved_ids = {path for path in moved_plastic_test}
    kept_test_ids = {path for path in kept_plastic_test}
    for path in train_plastic:
        if path in kept_test_ids:
            split = "test"
        else:
            split = "train"
        assigned.append((split, "plastic", "stage1", path, False))
    if moved_ids & kept_test_ids:
        raise SystemExit("Stage1 plastic test photos were also kept in train.")
    for path in val_plastic:
        if path.stem in locked["plastic"]:
            raise SystemExit(f"Val plastic stem is also locked test: {path.stem}")
        assigned.append(("validation", "plastic", "stage1", path, False))

    missing_test = locked["plastic"] - {path.stem for path in train_plastic}
    if missing_test:
        raise SystemExit(f"Missing {len(missing_test)} locked plastic test stems.")
    missing_metal = locked["metal"] - (
        {path.stem for path in test_metal}
        | {
            path.stem
            for path in image_files(STAGE1 / "train" / "NonPlastic")
            if "synth" in path.stem.lower()
        }
    )
    if missing_metal:
        raise SystemExit(f"Missing {len(missing_metal)} locked metal test stems.")

    return assigned, dropped


def belt_assignments():
    rng = random.Random(RANDOM_SEED)
    plastic_test = image_files(BELT / "test" / "plastic")
    kept_plastic_test, moved_plastic_test = sample_keep(
        plastic_test,
        BELT_PLASTIC_TEST_KEEP,
        rng,
    )
    print(
        f"Belt plastic test kept {len(kept_plastic_test)}, "
        f"moved {len(moved_plastic_test)} into train",
        flush=True,
    )
    kept_ids = {path for path in kept_plastic_test}
    assigned = []
    for split in ("train", "validation", "test"):
        for class_name in ("metal", "plastic"):
            for path in image_files(BELT / split / class_name):
                dest_split = split
                if split == "test" and class_name == "plastic" and path not in kept_ids:
                    dest_split = "train"
                assigned.append((dest_split, class_name, "belt", path, False))
    return assigned


def resize_one(job):
    source, destination = job
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(
            (LONG_SIDE, LONG_SIDE),
            Image.Resampling.BILINEAR,
        )
        image.save(
            destination,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
        )
    return destination


def copy_one(job):
    source, destination = job
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def write_and_check(jobs):
    if DEST.exists():
        shutil.rmtree(DEST)
    for split in ("train", "validation", "test"):
        for class_name in ("metal", "plastic"):
            (DEST / split / class_name).mkdir(parents=True)

    resize_jobs = [(source, dest) for source, dest, resize in jobs if resize]
    copy_jobs = [(source, dest) for source, dest, resize in jobs if not resize]
    workers = min(8, os.cpu_count() or 4)

    print(
        f"Resizing {len(resize_jobs)} new cans to {LONG_SIDE} and "
        f"copying {len(copy_jobs)} photos already at {LONG_SIDE}.",
        flush=True,
    )
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(resize_one, job) for job in resize_jobs]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done == 1 or done == len(resize_jobs) or done % 100 == 0:
                print(f"  resized {done}/{len(resize_jobs)}", flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_one, job) for job in copy_jobs]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done == 1 or done == len(copy_jobs) or done % 1000 == 0:
                print(f"  copied {done}/{len(copy_jobs)}", flush=True)


def file_md5(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(manifest_rows):
    stems = defaultdict(list)
    hashes = defaultdict(list)
    for row in manifest_rows:
        split, class_name, origin, name = row
        stems[Path(name).stem].append((split, class_name, origin))
        destination = DEST / split / class_name / name
        if not destination.is_file():
            raise SystemExit(f"Missing {destination}")
        with Image.open(destination) as image:
            if image.size != (LONG_SIDE, LONG_SIDE):
                raise SystemExit(f"{destination} is {image.size}, expected 224.")
        hashes[file_md5(destination)].append((split, class_name, name))

    stem_leaks = {stem: hits for stem, hits in stems.items() if len(hits) > 1}
    hash_leaks = {
        digest: hits
        for digest, hits in hashes.items()
        if len({hit[0] for hit in hits}) > 1 or len(hits) > 1
    }
    if stem_leaks or hash_leaks:
        print(f"stem collisions: {len(stem_leaks)}")
        print(f"byte collisions: {len(hash_leaks)}")
        for stem, hits in list(stem_leaks.items())[:5]:
            print(" ", stem, hits)
        raise SystemExit("Split leak: the same photo is in more than one place.")

    counts = Counter((row[0], row[1], row[2]) for row in manifest_rows)
    print()
    print(f"{'origin':<10} {'class':<8} {'train':>7} {'val':>7} {'test':>7}")
    for origin in ("stage1", "belt", "sep28", "sep29", "sep30"):
        for class_name in ("metal", "plastic"):
            print(
                f"{origin:<10} {class_name:<8} "
                f"{counts[('train', class_name, origin)]:7} "
                f"{counts[('validation', class_name, origin)]:7} "
                f"{counts[('test', class_name, origin)]:7}"
            )

    def total(split, origin_prefix, class_name=None):
        return sum(
            value
            for (split_name, klass, origin), value in counts.items()
            if split_name == split
            and origin.startswith(origin_prefix)
            and (class_name is None or klass == class_name)
        )

    print()
    for split in ("train", "validation", "test"):
        old = total(split, "stage1")
        new = total(split, "belt") + total(split, "sep")
        old_metal = total(split, "stage1", "metal")
        new_metal = total(split, "belt", "metal") + total(split, "sep", "metal")
        print(
            f"{split}: old {old} (metal {old_metal}), "
            f"newer {new} (metal {new_metal})"
        )
    train_old_metal = total("train", "stage1", "metal")
    train_new_metal = total("train", "belt", "metal") + total("train", "sep", "metal")
    print(
        f"train metal balance: old {train_old_metal}, "
        f"belt cans {train_new_metal}"
    )
    return counts


def capture_time(path):
    date = None
    for part in Path(path).stem.split("_"):
        if len(part) == 8 and part.isdigit() and part.startswith("202"):
            date = part
            continue
        if date and len(part) == 6 and part.isdigit():
            return datetime.strptime(date + part, "%Y%m%d%H%M%S")
    return None


def cluster_bursts(paths):
    paths = sorted(
        paths,
        key=lambda path: (capture_time(path) or datetime.min, path.name),
    )
    bursts = []
    current = [paths[0]]
    for path in paths[1:]:
        previous = capture_time(current[-1])
        this = capture_time(path)
        if (
            previous
            and this
            and (this - previous).total_seconds() <= BURST_GAP_SECONDS
        ):
            current.append(path)
        else:
            bursts.append(current)
            current = [path]
    bursts.append(current)
    return bursts


def choose_stage1_metal_moves(test_paths, locked_metal, move_n):
    """Stems of stage1 test metal that move to validation.

    test_paths includes locked stems so a burst that touches the locked
    test list stays in test. move_n is how many photos to move.
    """
    test_paths = list(test_paths)
    by_stem = {path.stem: path for path in test_paths}
    if len(by_stem) != len(test_paths):
        raise SystemExit("Stage1 test metal has duplicate stems.")
    missing = (SHEET_FAIL_TO_VAL | SHEET_FAIL_STAY_TEST) - set(by_stem)
    if missing:
        raise SystemExit(
            "Sheet failures missing from stage1 test metal: "
            + ", ".join(sorted(missing))
        )
    if move_n < len(SHEET_FAIL_TO_VAL):
        raise SystemExit(
            f"Need to move {move_n} stage1 metal photos, "
            f"but {len(SHEET_FAIL_TO_VAL)} sheet failures are pinned to validation."
        )

    move_stems = set()
    free = []
    for burst in cluster_bursts(test_paths):
        stems = {path.stem for path in burst}
        locked_hit = stems & set(locked_metal)
        stay_hit = stems & SHEET_FAIL_STAY_TEST
        val_hit = stems & SHEET_FAIL_TO_VAL
        if val_hit and (stay_hit or locked_hit):
            raise SystemExit(
                "A foil-tray failure pinned to validation shares a burst "
                "with a test hold: " + ", ".join(sorted(stems))
            )
        if val_hit:
            move_stems |= stems
        elif stay_hit or locked_hit:
            continue
        else:
            free.append(burst)

    need = move_n - len(move_stems)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(free)
    leftover = []
    for burst in free:
        if len(burst) <= need:
            move_stems.update(path.stem for path in burst)
            need -= len(burst)
        else:
            leftover.append(burst)
    if need:
        leftover.sort(key=lambda burst: (len(burst), burst[0].name))
        for path in leftover[0][:need]:
            move_stems.add(path.stem)
        need = 0

    val_sheets = SHEET_FAIL_TO_VAL & move_stems
    test_sheets = SHEET_FAIL_STAY_TEST - move_stems
    if not val_sheets or not test_sheets:
        raise SystemExit("Foil-tray failures landed on only one side.")
    if len(move_stems) != move_n:
        raise SystemExit(
            f"Moved {len(move_stems)} stage1 metal photos, wanted {move_n}."
        )
    return move_stems


def rebalance_stage1_metal(stage_rows, other_val_metal, other_test_metal):
    stage_val = sum(
        1
        for split, class_name, *_ in stage_rows
        if split == "validation" and class_name == "metal"
    )
    test_paths = [
        path
        for split, class_name, origin, path, _ in stage_rows
        if split == "test" and class_name == "metal" and origin == "stage1"
    ]
    val_metal = other_val_metal + stage_val
    test_metal = other_test_metal + len(test_paths)
    move_n = (test_metal - val_metal) // 2
    if move_n <= 0:
        return stage_rows
    locked = load_locked_test()["metal"]
    move_stems = choose_stage1_metal_moves(test_paths, locked, move_n)
    updated = []
    for row in stage_rows:
        split, class_name, origin, path, resize = row
        if path.stem in move_stems:
            if split != "test" or class_name != "metal":
                raise SystemExit(f"{path.name} was not stage1 test metal.")
            split = "validation"
        updated.append((split, class_name, origin, path, resize))
    print(
        f"Stage1 metal test -> validation: {len(move_stems)}. "
        f"Metal will be {val_metal + len(move_stems)} validation / "
        f"{test_metal - len(move_stems)} test. "
        f"Foil trays in validation: {', '.join(sorted(SHEET_FAIL_TO_VAL))}. "
        f"Foil trays left in test: {', '.join(sorted(SHEET_FAIL_STAY_TEST))}.",
        flush=True,
    )
    return updated


def write_zip():
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    files = sorted(
        path
        for path in DEST.rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    )
    print(f"Zipping {len(files)} files -> {ZIP_PATH}", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, path.relative_to(DEST).as_posix())
    print(f"Zip size {ZIP_PATH.stat().st_size / (1024 * 1024):.1f} MB", flush=True)


def main():
    for folder in (STAGE1, BELT, *(path for _, path in NEW_CAN_DIRS)):
        if not folder.is_dir():
            raise SystemExit(f"Missing {folder}")

    print("Sep 28-30 cans:", flush=True)
    new_rows = assign_new_cans()
    can_train = sum(1 for split, _, _ in new_rows if split == "train")
    belt_rows = belt_assignments()
    belt_metal_train = sum(
        1
        for split, class_name, _, _, _ in belt_rows
        if split == "train" and class_name == "metal"
    )
    can_train_total = can_train + belt_metal_train
    print(
        f"Belt cans in train: {belt_metal_train} existing + {can_train} new "
        f"= {can_train_total}. Stage1 NonPlastic train is thinned to that count.",
        flush=True,
    )

    stage_rows, dropped = stage1_assignments(can_train_total)
    print("Stage1 synth dropped, real extras to test:", dict(dropped), flush=True)
    other_val_metal = sum(
        1
        for split, class_name, *_ in belt_rows
        if split == "validation" and class_name == "metal"
    ) + sum(1 for split, _, _ in new_rows if split == "validation")
    other_test_metal = sum(
        1
        for split, class_name, *_ in belt_rows
        if split == "test" and class_name == "metal"
    ) + sum(1 for split, _, _ in new_rows if split == "test")
    stage_rows = rebalance_stage1_metal(
        stage_rows,
        other_val_metal,
        other_test_metal,
    )

    jobs = []
    manifest = []
    seen_names = {}
    for split, class_name, origin, path, resize in belt_rows + stage_rows:
        destination_name = path.name
        if destination_name in seen_names:
            raise SystemExit(
                f"{destination_name} comes from both "
                f"{seen_names[destination_name]} and {origin}."
            )
        seen_names[destination_name] = origin
        destination = DEST / split / class_name / destination_name
        jobs.append((path, destination, resize))
        manifest.append((split, class_name, origin, destination_name))

    for split, origin, path in new_rows:
        if split is None:
            continue
        destination_name = path.name
        if destination_name in seen_names:
            raise SystemExit(f"{destination_name} already assigned.")
        seen_names[destination_name] = origin
        destination = DEST / split / "metal" / destination_name
        jobs.append((path, destination, True))
        manifest.append((split, "metal", origin, destination_name))

    write_and_check(jobs)
    manifest_path = DEST / "SPLIT_MANIFEST.tsv"
    lines = ["split\tclass\torigin\tfilename"]
    lines.extend("\t".join(row) for row in manifest)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit(manifest)
    write_zip()
    print(f"Dataset: {DEST}", flush=True)


if __name__ == "__main__":
    main()
