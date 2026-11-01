from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import pandas as pd
from PIL import Image
import yaml

CLASS_NAMES = ['longitudinal_crack', 'transverse_crack', 'alligator_crack', 'pothole']
SPLITS = ('train', 'val', 'test', 'saudi_test')
EXPECTED_SIZES = {'train': (640,640), 'val': (640,640), 'test': (640,640), 'saudi_test': (640,640)}
EXTENSIONS = {'.jpg','.jpeg','.png','.bmp','.webp'}
RDD_NAMES = dict(enumerate(CLASS_NAMES))

def portable(value):
    return str(value).replace(chr(92), '/')

def names_for(split):
    return RDD_NAMES

# Read normalized class-and-box labels and convert valid boxes to pixel coordinates.
def read_boxes(path, size, names):
    width, height = size
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Line {number}: expected five class-and-box fields")
        cls = int(fields[0])
        x, y, w, h = map(float, fields[1:])
        if cls not in names or not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f"Line {number}: invalid class or non-finite coordinate")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Line {number}: invalid normalized coordinates")
        edges = (x-w/2, y-h/2, x+w/2, y+h/2)
        if min(edges) < -1e-5 or max(edges) > 1+1e-5:
            raise ValueError(f"Line {number}: box exceeds image boundaries")
        x1, y1, x2, y2 = [min(1, max(0, v)) for v in edges]
        box = (cls, x1*width, y1*height, x2*width, y2*height)
        if box in boxes:
            raise ValueError(f"Line {number}: duplicate box")
        boxes.append(box)
    return boxes


def validate_dataset(data_root, workers=8):
    DATA_ROOT = Path(data_root).resolve()
    WORKERS = workers
    version_info = json.loads((DATA_ROOT / "dataset_version.json").read_text(encoding="utf-8"))
    report = {"split_images": {s: int(v["images"]) for s, v in version_info["splits"].items()}}
    integrity = json.loads((DATA_ROOT / "integrity_sha256.json").read_text(encoding="utf-8"))
    metadata = {}
    for line in (DATA_ROOT / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = portable(row['image'])
        if key in metadata or Path(key).is_absolute() or '..' in Path(key).parts:
            raise ValueError(f'Duplicate or invalid manifest image: {key}')
        metadata[key] = row
    def inspect(task):
        split, image_path, label_path = task
        relative = image_path.relative_to(DATA_ROOT).as_posix()
        try:
            row = metadata[relative]
            domain = "local" if split == "saudi_test" else "rdd"
            if row["split"] != split or row["domain"] != domain:
                raise ValueError("Manifest split or domain mismatch")
            if portable(row["label"]) != label_path.relative_to(DATA_ROOT).as_posix():
                raise ValueError("Manifest label path mismatch")
            with Image.open(image_path) as image:
                image.load()
                if image.mode != "RGB" or image.size != EXPECTED_SIZES[split]:
                    raise ValueError("Unexpected prepared image mode or dimensions for this split")
                if image.getexif().get(274, 1) != 1:
                    raise ValueError("Unapplied EXIF orientation")
                digest = hashlib.sha256(f"{image.width},{image.height}".encode() + image.tobytes()).hexdigest()
                boxes = read_boxes(label_path, image.size, names_for(split))
            if digest != row["export_pixel_sha256"]:
                raise ValueError("Image pixels differ from preparation metadata")
            if [sum(b[0] == c for b in boxes) for c in names_for(split)] != row["class_counts"]:
                raise ValueError("Label counts differ from preparation metadata")
            if "known_annotation_classes" in row and row["known_annotation_classes"] != list(names_for(split)):
                raise ValueError("Manifest annotation schema mismatch")
            for member in (image_path, label_path):
                key = member.relative_to(DATA_ROOT).as_posix()
                if key not in integrity or hashlib.sha256(member.read_bytes()).hexdigest() != integrity[key]:
                    raise ValueError(f"Approved file fingerprint mismatch: {key}")
            return dict(split=split, relative=relative, image_path=image_path,
                        boxes=boxes, sha256=digest, group=row.get("scene_group")), None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return None, dict(path=relative, issue=str(exc))

    # Build paired-file tasks and check directory consistency.
    tasks, errors = [], []
    found = defaultdict(int)
    seen_images = set()
    for split in SPLITS:
        image_dir, label_dir = DATA_ROOT / "images" / split, DATA_ROOT / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(dict(path=split, issue="Missing images or labels directory"))
        images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        found[split] = len(images)
        stems, expected_labels = set(), set()
        for path in images:
            stem = path.relative_to(image_dir).with_suffix("").as_posix()
            if stem in stems:
                errors.append(dict(path=str(path), issue="Duplicate image stem"))
            stems.add(stem)
            label = label_dir / path.relative_to(image_dir).with_suffix(".txt")
            expected_labels.add(label)
            seen_images.add(path.relative_to(DATA_ROOT).as_posix())
            tasks.append((split, path, label))
        for label in label_dir.rglob("*.txt"):
            if label not in expected_labels:
                errors.append(dict(path=str(label), issue="Label without image"))
    for missing in sorted(set(metadata) - seen_images):
        errors.append(dict(path=missing, issue="Manifest image missing from disk"))

    # Decode images and validate annotations in parallel.
    records = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for index, (record, error) in enumerate(pool.map(inspect, tasks), 1):
            if record is not None:
                records.append(record)
            if error is not None:
                errors.append(error)
            if index % 2000 == 0 or index == len(tasks):
                print(f"Checked {index}/{len(tasks)} images")

    # Check exact duplicates and overlap of available scene groups.
    hashes, groups = {}, defaultdict(set)
    for record in records:
        digest = record["sha256"]
        if digest in hashes:
            errors.append(dict(path=record["relative"], issue="Exact duplicate of " + hashes[digest]))
        hashes[digest] = record["relative"]
        if record["group"]:
            groups[record["group"]].add(record["split"])
    for group, splits in groups.items():
        if len(splits) > 1:
            errors.append(dict(path=str(group), issue=f"Group overlaps splits: {sorted(splits)}"))

    # Summarize pairs, boxes and empty labels for each split.
    summary_rows, class_rows = [], []
    for split in SPLITS:
        subset = [r for r in records if r["split"] == split]
        expected = report["split_images"][split]
        if len(subset) != expected:
            errors.append(dict(path=split, issue=f"Expected {expected} valid pairs, found {len(subset)}"))
        summary_rows.append(dict(split=split, images_found=found[split], valid_pairs=len(subset),
                                 boxes=sum(len(r["boxes"]) for r in subset),
                                 empty_labels=sum(not r["boxes"] for r in subset)))
        for cls, name in names_for(split).items():
            boxes = sum(b[0] == cls for r in subset for b in r["boxes"])
            class_rows.append(dict(split=split, class_name=name,
                                   images=sum(any(b[0] == cls for b in r["boxes"]) for r in subset), boxes=boxes))
            if split != "saudi_test" and boxes == 0:
                errors.append(dict(path=split, issue=f"Missing RDD class: {name}"))

    if errors:
        raise ValueError('Dataset validation failed: ' + json.dumps(errors[:30], ensure_ascii=False))
    if not records:
        raise ValueError('Dataset contains no valid images')
    return {'summary': summary_rows, 'classes': class_rows, 'errors': [],
            'note': 'Structural checks do not certify annotation completeness or physical scene independence.'}

def list_images(data_root, split):
    return sorted(p for p in (Path(data_root)/'images'/split).rglob('*')
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)

def read_label_ids(data_root, image_path, split):
    image_path = Path(image_path)
    relative = image_path.relative_to(Path(data_root)/'images'/split).with_suffix('.txt')
    label = Path(data_root)/'labels'/split/relative
    if not label.is_file():
        raise FileNotFoundError(f'Missing annotation: {label}')
    with Image.open(image_path) as image:
        return [box[0] for box in read_boxes(label, image.size, RDD_NAMES)]

def make_items(data_root, split, include_confirmed_empty_local=False):
    import numpy as np
    items, excluded = [], []
    for image in list_images(data_root, split):
        ids = read_label_ids(data_root, image, split)
        if split == 'saudi_test' and not ids and not include_confirmed_empty_local:
            excluded.append(str(image))
            continue
        y = np.zeros(4, dtype=np.float32)
        y[list(set(ids))] = 1
        items.append((str(image), y))
    return items, excluded
