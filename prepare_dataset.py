"""
prepare_dataset.py
Finds the 3 tomato classes in PlantVillage, picks 100 images each,
and splits them into train / val / test (70 / 15 / 15).

Usage:
    python prepare_dataset.py --src "C:/path/to/PlantVillage"
"""

import os
import shutil
import random
import argparse

CLASSES = {
    "healthy":      "Tomato_healthy",
    "early_blight": "Tomato_Early_blight",
    "late_blight":  "Tomato_Late_blight",
}
IMAGES_PER_CLASS = 100
SPLITS = {"train": 70, "val": 15, "test": 15}   # must sum to 100
SEED = 42
OUTPUT_DIR = "data"


def find_folder(root, keyword):
    """Find a folder inside root whose name contains keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for name in os.listdir(root):
        if keyword_lower in name.lower() and os.path.isdir(os.path.join(root, name)):
            return os.path.join(root, name)
    return None


def main(src_root):
    random.seed(SEED)

    print(f"\n Source: {src_root}")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}\n")

    for class_label, keyword in CLASSES.items():
        folder = find_folder(src_root, keyword)
        if folder is None:
            print(f" Could not find folder for '{keyword}' in {src_root}")
            print(f" Folders found: {os.listdir(src_root)}")
            return

        all_imgs = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(all_imgs) < IMAGES_PER_CLASS:
            print(f"Only {len(all_imgs)} images found for '{class_label}', using all.")
            selected = all_imgs
        else:
            selected = random.sample(all_imgs, IMAGES_PER_CLASS)

        random.shuffle(selected)
        splits = {
            "train": selected[:SPLITS["train"]],
            "val":   selected[SPLITS["train"]: SPLITS["train"] + SPLITS["val"]],
            "test":  selected[SPLITS["train"] + SPLITS["val"]:],
        }

        for split_name, files in splits.items():
            dest = os.path.join(OUTPUT_DIR, split_name, class_label)
            os.makedirs(dest, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(folder, f), os.path.join(dest, f))

        print(f"{class_label:15s} → train:{len(splits['train'])}  val:{len(splits['val'])}  test:{len(splits['test'])}")

    print("\n Dataset ready! Folder structure:")
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            path = os.path.join(OUTPUT_DIR, split, cls)
            count = len(os.listdir(path)) if os.path.exists(path) else 0
            print(f"   {path:35s}  {count} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to the extracted PlantVillage folder")
    args = parser.parse_args()
    main(args.src)
