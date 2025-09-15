import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def find_ta_root(repo_root: Path) -> Path:
    cand = repo_root / "TA"
    if (cand / "Healthy" / "Images").is_dir() and (cand / "Healthy" / "Masks").is_dir():
        return cand
    raise FileNotFoundError("TA dataset folder not found under repo root")


def nnunet_raw_base() -> Path:
    p = os.environ.get("nnUNet_raw")
    if p:
        return Path(p)
    # fallback to common path used earlier
    return Path("/mnt/e/nnUNet/nnUNet_raw")


def convert_split(split_root: Path, out_images: Path, out_labels: Path, prefix: str) -> int:
    images_dir = split_root / "Images"
    masks_dir = split_root / "Masks"
    assert images_dir.is_dir() and masks_dir.is_dir(), f"Invalid TA split structure at {split_root}"

    n = 0
    for img_path in sorted(images_dir.glob("*.png")):
        base = img_path.stem  # e.g., anon_1001_1
        msk_path = masks_dir / f"{base}.png"
        if not msk_path.is_file():
            raise FileNotFoundError(f"Missing mask for {img_path} -> {msk_path}")

        case_id = f"TA{prefix}_{base}"

        # copy image as single-channel input -> _0000.png
        with Image.open(img_path) as im:
            # ensure grayscale 'L'
            if im.mode != "L":
                im = im.convert("L")
            out_img = out_images / f"{case_id}_0000.png"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            im.save(out_img)

        # load mask, map 255->1, keep 0 as 0
        with Image.open(msk_path) as m:
            if m.mode != "L":
                m = m.convert("L")
            arr = np.array(m)
            # binary to {0,1}
            arr = (arr > 0).astype(np.uint8)
            out_msk = out_labels / f"{case_id}.png"
            out_msk.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr, mode="L").save(out_msk)

        n += 1
    return n


def generate_dataset_json(dataset_root: Path, num_training: int):
    # late import to avoid hard dependency if script is inspected elsewhere
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

    channel_names = {0: "image"}
    labels = {"background": 0, "target": 1}
    generate_dataset_json(
        output_folder=str(dataset_root),
        channel_names=channel_names,
        labels=labels,
        num_training_cases=num_training,
        file_ending=".png",
        dataset_name="Dataset501_TA",
        description="TA 2D grayscale PNG with binary masks",
        overwrite_image_reader_writer="NaturalImage2DIO",
    )


def main():
    repo_root = Path(__file__).resolve().parents[1]
    ta_root = find_ta_root(repo_root)
    out_base = nnunet_raw_base() / "Dataset501_TA"
    imagesTr = out_base / "imagesTr"
    labelsTr = out_base / "labelsTr"
    imagesTs = out_base / "imagesTs"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)
    imagesTs.mkdir(parents=True, exist_ok=True)

    total = 0
    total += convert_split(ta_root / "Healthy", imagesTr, labelsTr, prefix="H")
    total += convert_split(ta_root / "Pathological", imagesTr, labelsTr, prefix="P")

    generate_dataset_json(out_base, total)

    print(f"Converted TA dataset -> {out_base}")
    print(f"Training cases: {total}")


if __name__ == "__main__":
    main()

