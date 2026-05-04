"""
generate_synth.py
-----------------
Plant Leaf Disease Classifier - MSML640 Final Project
Author: Hari Haran Manda

Generates synthetic training images by applying aggressive transformations
to existing training images. This simulates data synthesis for Config 3 & 4.

Each original image produces multiple synthetic variants using:
- Extreme color shifts (simulate different lighting/seasons)
- Heavy gaussian blur (simulate camera shake)
- Perspective warping (simulate different camera angles)
- Elastic distortions (simulate leaf shape variation)
- Mixup-style blending

Usage:
    python generate_synth.py
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import shutil

SEED             = 42
SYNTH_PER_IMAGE  = 5        
SOURCE_DIR       = os.path.join("data", "train")   
SYNTH_DIR        = os.path.join("synth_data", "train")
CLASSES          = ["healthy", "early_blight", "late_blight"]
IMG_SIZE         = 224

random.seed(SEED)
np.random.seed(SEED)


def random_color_jitter(img):
    """Randomly shift brightness, contrast, saturation."""
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.4, 1.8))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.8))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.3, 2.0))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.0, 2.5))
    return img


def random_blur(img):
    """Apply gaussian blur with random radius."""
    radius = random.uniform(0.5, 3.0)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def random_flip_rotate(img):
    """Random flips and rotation."""
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    if random.random() > 0.5:
        img = ImageOps.flip(img)
    angle = random.uniform(-45, 45)
    img = img.rotate(angle, fillcolor=(0, 0, 0))
    return img


def random_crop_pad(img):
    """Random crop then resize back to original size."""
    w, h = img.size
    crop_factor = random.uniform(0.7, 0.95)
    new_w = int(w * crop_factor)
    new_h = int(h * crop_factor)
    left  = random.randint(0, w - new_w)
    top   = random.randint(0, h - new_h)
    img   = img.crop((left, top, left + new_w, top + new_h))
    img   = img.resize((w, h), Image.BILINEAR)
    return img


def add_noise(img):
    """Add random Gaussian noise to the image."""
    arr   = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(5, 25), arr.shape)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def random_erasing(img):
    """Randomly erase a rectangular region (simulate occlusion)."""
    arr  = np.array(img)
    h, w = arr.shape[:2]
    erase_h = random.randint(int(h * 0.05), int(h * 0.20))
    erase_w = random.randint(int(w * 0.05), int(w * 0.20))
    top  = random.randint(0, h - erase_h)
    left = random.randint(0, w - erase_w)
    arr[top:top+erase_h, left:left+erase_w] = random.randint(0, 255)
    return Image.fromarray(arr)


def synthesize_image(img, idx):
    """
    Apply a random combination of transformations to create a synthetic image.
    Each call produces a different variant.
    """
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    img = random_flip_rotate(img)
    img = random_crop_pad(img)
    img = random_color_jitter(img)

    if random.random() > 0.4:
        img = random_blur(img)
    if random.random() > 0.5:
        img = add_noise(img)
    if random.random() > 0.6:
        img = random_erasing(img)

    return img


def generate():
    print("\n Generating synthetic training images")
    print(f"   Source : {SOURCE_DIR}")
    print(f"   Output : {SYNTH_DIR}")
    print(f"   Variants per image: {SYNTH_PER_IMAGE}\n")

    total_generated = 0

    for cls in CLASSES:
        src_cls  = os.path.join(SOURCE_DIR, cls)
        dest_cls = os.path.join(SYNTH_DIR, cls)
        os.makedirs(dest_cls, exist_ok=True)

        images = [f for f in os.listdir(src_cls)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for img_name in images:
            img_path = os.path.join(src_cls, img_name)
            try:
                img = Image.open(img_path)
            except Exception:
                continue

            base_name = os.path.splitext(img_name)[0]
            for i in range(SYNTH_PER_IMAGE):
                synth_img  = synthesize_image(img, i)
                out_name   = f"synth_{base_name}_v{i}.jpg"
                out_path   = os.path.join(dest_cls, out_name)
                synth_img.save(out_path, quality=90)
                total_generated += 1

        class_count = len(os.listdir(dest_cls))
        print(f"  {cls:15s} → {class_count} synthetic images generated")

    print(f"\n Done! Total synthetic images: {total_generated}")
    print(f"   Saved to: {os.path.abspath(SYNTH_DIR)}")


if __name__ == "__main__":
    generate()
