"""Belt-check numbers for one photo.

The live area is the same mask as presence.foreground_parts. The other
numbers are the measurements from the belt-check canvas: the second
piece, chroma, brightness with the rail taken out, the 224-square shape
checks, and the blank-spectrum distances.

Each function returns a raw score. The cut lives on the pipeline card,
so moving it does not read the file again.
"""

import cv2
import numpy as np

from presence import PresenceError, foreground_parts


PIECE_FLOOR = 20
CHROMA_LEVEL = 12
CHROMA_WIDTH = 224
SQUARE = 224
SQUARE_DEVIATION = 18
SLOT_AREA = 80
SLOT_ASPECT = 2.5
SLOT_VERTICAL = 0.7
TALL_ASPECT = 3.5
TALL_VERTICAL = 0.75
BRIGHT_WIDTH = 224
NEAREST_K = 5

# Filename pieces left out of the empty-belt spectrum in the fair trial.
AUDIT_SKIP = ("03497", "00000_20260710")


CHECKS = (
    {
        "id": "largest_blob",
        "label": "Largest blob",
        "empty_when": "below",
        "cut": 159,
        "fixed": True,
        "detail": "Width-64 mask, 28 from the median. Empty when the largest piece is under 159.",
    },
    {
        "id": "piece_sum",
        "label": "Pieces combined",
        "empty_when": "below",
        "cut": 159,
        "fixed": True,
        "detail": "Largest piece plus every other piece of at least 20 pixels on that same mask.",
    },
    {
        "id": "second_piece",
        "label": "Second piece",
        "empty_when": "below",
        "cut": 80,
        "fixed": False,
        "detail": "The second piece, when it is at least 20 pixels. On a split can this is the can, and the largest piece is the bottom band.",
    },
    {
        "id": "chroma",
        "label": "Chroma",
        "empty_when": "below",
        "cut": 0.02,
        "fixed": True,
        "detail": "Share of pixels whose Lab chroma is at least 12, on a width-224 view. Empty under 2%.",
    },
    {
        "id": "brightness",
        "label": "Brightness",
        "empty_when": "below",
        "cut": 200,
        "fixed": False,
        "detail": "Largest bright piece after the bottom metal strip and the wide short band above it are taken out.",
    },
    {
        "id": "aspect",
        "label": "Aspect",
        "empty_when": "above",
        "cut": 3.5,
        "fixed": False,
        "detail": "Long side over short side of the largest piece on a forced 224 square. A tall slot scores high.",
    },
    {
        "id": "largest_224",
        "label": "Largest blob, 224",
        "empty_when": "below",
        "cut": 40,
        "fixed": False,
        "detail": "Largest piece after every photo is forced to 224×224. Deviation 18. Not the live area.",
    },
    {
        "id": "drop_tall",
        "label": "Drop tall slots",
        "empty_when": "below",
        "cut": 40,
        "fixed": False,
        "detail": "That 224 area, unless the largest piece itself is a tall slot. Then the score is 0.",
    },
    {
        "id": "fourier",
        "label": "Average pattern",
        "empty_when": "below",
        "cut": 0.004,
        "fixed": True,
        "detail": "Cosine distance from the log Fourier magnitude to the mean of the audited empty belts.",
    },
    {
        "id": "nearest_blanks",
        "label": "Nearest empty belts",
        "empty_when": "below",
        "cut": 0.02,
        "fixed": False,
        "detail": "Mean cosine distance to the five nearest audited empty-belt spectra.",
    },
)


def check_by_id(check_id):
    for item in CHECKS:
        if item["id"] == check_id:
            return item
    return None


def _round(value):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, 6)


def audited_blank(path):
    name = path.name
    return not any(piece in name for piece in AUDIT_SKIP)


def _read_color(path):
    color = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if color is None:
        return None
    return color


def _square_norm(gray):
    image = cv2.resize(gray, (SQUARE, SQUARE), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    sigma = SQUARE / 40.0
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(image / np.maximum(blur, 1.0) * 255.0, 0, 255)


def spectrum_of(image):
    centered = image - float(image.mean())
    magnitude = np.log1p(np.abs(np.fft.fft2(centered))).astype(np.float32)
    magnitude[0, 0] = 0
    flat = magnitude.ravel()
    norm = float(np.linalg.norm(flat))
    if norm < 1e-6:
        return None
    return flat / norm


def _shape(labels, index):
    ys, xs = np.where(labels == index)
    if len(xs) < 5:
        return 1.0, 0.0
    points = np.stack([xs, ys], 1).astype(np.float32)
    rect = cv2.minAreaRect(points)
    width, height = [max(float(value), 1e-3) for value in rect[1]]
    angle = rect[2]
    if width < height:
        angle += 90
    angle = abs(angle) % 180
    if angle > 90:
        angle = 180 - angle
    aspect = max(width, height) / min(width, height)
    vertical = 1.0 - min(abs(angle - 90.0), 90.0) / 90.0
    return float(aspect), float(vertical)


def square_scores(gray):
    image = _square_norm(gray)
    median = float(np.median(image))
    mask = (np.abs(image - median) >= SQUARE_DEVIATION).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    spec = spectrum_of(image)
    if count <= 1:
        return {"largest_224": 0, "aspect": 1.0, "drop_tall": 0}, spec
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32)
    aspects = []
    verticals = []
    for index in range(1, count):
        aspect, vertical = _shape(labels, index)
        aspects.append(aspect)
        verticals.append(vertical)
    aspects = np.asarray(aspects, dtype=np.float64)
    verticals = np.asarray(verticals, dtype=np.float64)
    chosen = int(np.argmax(areas))
    area = int(areas[chosen])
    slot = (areas <= SLOT_AREA) & (aspects >= SLOT_ASPECT) & (verticals >= SLOT_VERTICAL)
    tall = aspects[chosen] >= TALL_ASPECT and verticals[chosen] >= TALL_VERTICAL
    drop_tall = 0 if slot[chosen] or tall else area
    return {
        "largest_224": area,
        "aspect": float(aspects[chosen]),
        "drop_tall": int(drop_tall),
    }, spec


def chroma_fraction(color):
    height, width = color.shape[:2]
    target = CHROMA_WIDTH if width > CHROMA_WIDTH else width
    resized_height = max(1, int(round(height * target / float(width))))
    if target != width:
        small = cv2.resize(
            color,
            (target, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = color
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
    return float((chroma >= CHROMA_LEVEL).mean())


def brightness_rest(color):
    height, width = color.shape[:2]
    target = BRIGHT_WIDTH if width > BRIGHT_WIDTH else width
    resized_height = max(1, int(round(height * target / float(width))))
    if target != width:
        small = cv2.resize(
            color,
            (target, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = color
    frame_h, frame_w = small.shape[:2]
    lightness = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    median = float(np.median(lightness))
    mask = (lightness >= max(median + 40.0, 90.0)).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    best = 0
    for index in range(1, count):
        _x, y, blob_w, blob_h, area = [int(value) for value in stats[index]]
        touches = (y + blob_h) >= frame_h - 3
        rail_h = max(5, int(round(0.035 * frame_h)))
        is_rail = (
            touches
            and blob_w >= 0.35 * frame_w
            and blob_h <= rail_h
            and blob_w > 3 * max(blob_h, 1)
        )
        is_lip = (
            not is_rail
            and blob_w >= 0.35 * frame_w
            and blob_h <= 0.16 * frame_h
            and blob_w > 3 * max(blob_h, 1)
        )
        if not is_rail and not is_lip and area > best:
            best = area
    return int(best)


def live_scores(path):
    try:
        parts = foreground_parts(path)
    except PresenceError:
        return None
    largest = parts[0] if parts else 0
    second = parts[1] if len(parts) > 1 and parts[1] >= PIECE_FLOOR else 0
    piece_sum = sum(area for area in parts if area >= PIECE_FLOOR)
    return {
        "largest_blob": int(largest),
        "second_piece": int(second),
        "piece_sum": int(piece_sum),
    }


def measure_image(path):
    """Raw check scores and the 224 spectrum, or None if the file will not open."""
    color = _read_color(path)
    if color is None:
        return None, None
    live = live_scores(path)
    if live is None:
        return None, None
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    square, spec = square_scores(gray)
    scores = dict(live)
    scores.update(square)
    scores["chroma"] = chroma_fraction(color)
    scores["brightness"] = brightness_rest(color)
    return scores, spec


def pattern_distance(spec, mean, bank):
    if spec is None or mean is None or bank is None or len(bank) == 0:
        return None, None
    fourier = float(1.0 - np.dot(spec, mean))
    sims = bank @ spec
    k = min(NEAREST_K, sims.shape[0])
    if k < sims.shape[0]:
        chosen = np.partition(sims, -k)[-k:]
    else:
        chosen = sims
    nearest = float(1.0 - chosen.mean())
    return fourier, nearest


def suggest_cut(blank_values, object_values, empty_when):
    """Highest blank catch that holds at most 1% of the object photos.

    ``empty_when`` is ``below`` or ``above``. Returns the cut, or None
    when no cut stays inside that hold.
    """
    blanks = np.asarray(list(blank_values), dtype=np.float64)
    objects = np.asarray(list(object_values), dtype=np.float64)
    blanks = blanks[np.isfinite(blanks)]
    objects = objects[np.isfinite(objects)]
    if blanks.size == 0 or objects.size == 0:
        return None
    levels = np.unique(np.concatenate([blanks, objects]))
    best = None
    for level in levels:
        if empty_when == "below":
            cut = float(level) + 1e-9
            caught = int(np.sum(blanks < cut))
            held = int(np.sum(objects < cut))
        else:
            cut = float(level)
            caught = int(np.sum(blanks > cut))
            held = int(np.sum(objects > cut))
        hold = held / float(objects.size)
        if hold > 0.01 + 1e-12:
            continue
        rank = (caught / float(blanks.size), -hold)
        if best is None or rank > best[0]:
            best = (rank, float(cut))
    if best is None:
        return None
    return best[1]
