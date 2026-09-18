from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "synthetic_yolo"

IMAGE_SIZE = 640
TRAIN_IMAGES = 512
VAL_IMAGES = 32
SEED = 20260817


def generate_sample(
    image_path: Path,
    label_path: Path,
    index: int,
    split: str,
    seed: int,
) -> None:
    """
    Generate one deterministic synthetic YOLO detection sample.

    Every image contains exactly one object and one YOLO-format label:
        class_id x_center y_center width height
    """

    split_offset = 0 if split == "train" else 1_000_000
    rng = random.Random(seed + split_offset + index)

    # Simple background keeps the dataset compact on disk.
    bg = (
        rng.randint(20, 80),
        rng.randint(20, 80),
        rng.randint(20, 80),
    )

    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), bg)
    draw = ImageDraw.Draw(image)

    # Generate exactly one object per image.
    box_w = rng.uniform(0.15, 0.35)
    box_h = rng.uniform(0.15, 0.35)

    x_center = rng.uniform(box_w / 2, 1.0 - box_w / 2)
    y_center = rng.uniform(box_h / 2, 1.0 - box_h / 2)

    x1 = int((x_center - box_w / 2) * IMAGE_SIZE)
    y1 = int((y_center - box_h / 2) * IMAGE_SIZE)
    x2 = int((x_center + box_w / 2) * IMAGE_SIZE)
    y2 = int((y_center + box_h / 2) * IMAGE_SIZE)

    color = (
        rng.randint(150, 255),
        rng.randint(150, 255),
        rng.randint(150, 255),
    )

    draw.rectangle(
        [x1, y1, x2, y2],
        fill=color,
        outline=(255, 255, 255),
        width=2,
    )

    # PNG is lossless and highly compressible for these simple images.
    image.save(image_path, format="PNG", optimize=True)

    # YOLO normalized label:
    # class x_center y_center width height
    label_path.write_text(
        f"0 {x_center:.8f} {y_center:.8f} {box_w:.8f} {box_h:.8f}\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()


def build_dataset_hash(dataset_root: Path) -> str:
    """
    Build a platform-independent semantic dataset hash.

    Images are hashed using decoded RGB pixel values rather than
    compressed PNG bytes, because PNG encoding may differ between
    Pillow/libpng versions while representing identical pixels.

    Label text is normalized to LF line endings.
    """

    hasher = hashlib.sha256()

    image_files = sorted(
        (dataset_root / "images").rglob("*.png")
    )

    label_files = sorted(
        (dataset_root / "labels").rglob("*.txt")
    )

    for path in image_files:
        relative = path.relative_to(
            dataset_root
        ).as_posix()

        hasher.update(
            relative.encode("utf-8")
        )

        with Image.open(path) as image:
            image = image.convert("RGB")

            hasher.update(
                f"{image.width}x{image.height}:RGB".encode(
                    "ascii"
                )
            )

            hasher.update(
                image.tobytes()
            )

    for path in label_files:
        relative = path.relative_to(
            dataset_root
        ).as_posix()

        hasher.update(
            relative.encode("utf-8")
        )

        text = path.read_text(
            encoding="utf-8"
        )

        canonical_text = (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        hasher.update(
            canonical_text.encode("utf-8")
        )

    return hasher.hexdigest()


def generate_dataset(dataset_root: Path, force: bool) -> None:
    if dataset_root.exists():
        existing_files = list(dataset_root.rglob("*"))

        if existing_files and not force:
            raise SystemExit(
                f"Dataset directory already exists and is not empty:\n"
                f"  {dataset_root}\n\n"
                f"Use --force only if you intentionally want to regenerate it."
            )

        if force:
            shutil.rmtree(dataset_root)

    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    train_labels = dataset_root / "labels" / "train"
    val_labels = dataset_root / "labels" / "val"

    for directory in (
        train_images,
        val_images,
        train_labels,
        val_labels,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    print("Generating training images...")

    for i in range(TRAIN_IMAGES):
        generate_sample(
            train_images / f"{i:06d}.png",
            train_labels / f"{i:06d}.txt",
            i,
            "train",
            SEED,
        )

    print("Generating validation images...")

    for i in range(VAL_IMAGES):
        generate_sample(
            val_images / f"{i:06d}.png",
            val_labels / f"{i:06d}.txt",
            i,
            "val",
            SEED,
        )

    # Use an absolute path generated locally on each machine.
    yaml_path = dataset_root.resolve().as_posix()

    data_yaml = (
        f'path: "{yaml_path}"\n'
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: synthetic_object\n"
    )

    (dataset_root / "data.yaml").write_text(
        data_yaml,
        encoding="utf-8",
    )

    dataset_hash = build_dataset_hash(dataset_root)

    manifest = {
        "dataset": "synthetic_yolo_runtime_prediction_v1",
        "seed": SEED,
        "image_width": IMAGE_SIZE,
        "image_height": IMAGE_SIZE,
        "train_images": TRAIN_IMAGES,
        "val_images": VAL_IMAGES,
        "objects_per_image": 1,
        "classes": 1,
        "class_names": ["synthetic_object"],
        "dataset_sha256": dataset_hash,
    }

    (dataset_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("Dataset generation complete")
    print(f"Dataset root : {dataset_root}")
    print(f"Train images : {TRAIN_IMAGES}")
    print(f"Val images   : {VAL_IMAGES}")
    print(f"Image size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Seed         : {SEED}")
    print(f"SHA256       : {dataset_hash}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and regenerate an existing dataset.",
    )

    args = parser.parse_args()

    generate_dataset(
        args.output.resolve(),
        args.force,
    )


if __name__ == "__main__":
    main()
