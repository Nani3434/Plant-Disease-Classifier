"""
analysis.py
-----------
Plant Leaf Disease Classifier - MSML640 Final Project
Author: Hari Haran Manda

Generates:
  1. Side-by-side confusion matrix comparison (Config 1 vs Config 3)
  2. Summary bar chart of all 4 configs
  3. Robustness testing under noise, blur, occlusion, brightness shifts
  4. Error analysis - shows misclassified images

Usage:
    python analysis.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageFilter
import warnings
warnings.filterwarnings("ignore")

CLASSES     = ["healthy", "early_blight", "late_blight"]
NUM_CLASSES = 3
IMG_SIZE    = 224
BATCH_SIZE  = 16
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"
DATA_DIR    = "data"
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
os.makedirs(RESULTS_DIR, exist_ok=True)

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_model(config_id):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES)
    )
    path = os.path.join(RESULTS_DIR, f"config{config_id}_model.pth")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def get_predictions(model, loader):
    all_preds, all_labels, all_imgs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_imgs.extend(imgs.cpu())
    return all_preds, all_labels, all_imgs


def make_cm(preds, labels):
    return confusion_matrix(labels, preds)


def plot_side_by_side_cm(cm1, cm3):
    """Side-by-side confusion matrices: Config 1 (baseline) vs Config 3 (best)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    titles = ["Config 1 — Baseline (73.33%)", "Config 3 — Synthesized (84.44%) Best"]
    cms    = [cm1, cm3]
    cmaps  = ['Blues', 'Greens']

    for ax, cm, title, cmap in zip(axes, cms, titles, cmaps):
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=15)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASSES, rotation=30, ha='right')
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(title, fontweight='bold')
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > 8 else 'black', fontsize=13)

    plt.suptitle('Baseline vs Best Configuration — Confusion Matrix Comparison',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparison_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_bar():
    """Bar chart of all 4 config test accuracies."""
    configs = ["Config 1\nBaseline", "Config 2\nAugmented",
               "Config 3\nSynthesized", "Config 4\nFull Pipeline"]
    accs    = [73.33, 73.33, 84.44, 75.56]
    colors  = ['#90CAF9', '#A5D6A7', '#388E3C', '#FF8A65']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(configs, accs, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
    ax.set_ylim(60, 100)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy Across All 4 Configurations', fontsize=13, fontweight='bold')
    ax.axhline(y=84.44, color='#388E3C', linestyle='--', alpha=0.6, linewidth=1.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.text(2, 85.5, 'Best', ha='center', color='#388E3C', fontweight='bold', fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "summary_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def apply_perturbation(img_pil, ptype):
    """
    Apply a specific perturbation to a PIL image.
    ptype options: 'gaussian_noise', 'motion_blur', 'occlusion', 'brightness'
    """
    img = img_pil.convert("RGB")

    if ptype == "gaussian_noise":
        arr   = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 40, arr.shape)
        arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif ptype == "motion_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=4))

    elif ptype == "occlusion":
        arr  = np.array(img)
        h, w = arr.shape[:2]
        top  = h // 4
        left = w // 4
        arr[top: top + h // 2, left: left + w // 2] = 0
        return Image.fromarray(arr)

    elif ptype == "brightness":
        from PIL import ImageEnhance
        return ImageEnhance.Brightness(img).enhance(0.3)

    return img


def robustness_test(model):
    """
    Evaluate best model (Config 3) under 4 perturbations.
    Returns dict: {perturbation: accuracy}
    """
    perturbations = ["clean", "gaussian_noise", "motion_blur", "occlusion", "brightness"]
    results       = {}

    to_tensor = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"))
    all_paths    = [s[0] for s in test_dataset.samples]
    all_labels   = [s[1] for s in test_dataset.samples]

    for ptype in perturbations:
        correct = 0
        with torch.no_grad():
            for path, label in zip(all_paths, all_labels):
                img = Image.open(path).convert("RGB")
                if ptype != "clean":
                    img = apply_perturbation(img, ptype)
                tensor = to_tensor(img).unsqueeze(0).to(DEVICE)
                output = model(tensor)
                pred   = output.argmax(dim=1).item()
                if pred == label:
                    correct += 1
        acc = correct / len(all_labels) * 100
        results[ptype] = acc
        print(f"  {ptype:20s}: {acc:.2f}%")

    return results


def plot_robustness(results):
    """Bar chart of robustness test results."""
    labels = list(results.keys())
    accs   = list(results.values())
    colors = ['#388E3C' if l == 'clean' else '#EF5350' for l in labels]
    display_labels = ["Clean\n(baseline)", "Gaussian\nNoise", "Motion\nBlur",
                      "Occlusion", "Low\nBrightness"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(display_labels, accs, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Robustness Testing — Config 3 (Best Model) Under Perturbations',
                 fontsize=12, fontweight='bold')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.axhline(y=accs[0], color='#388E3C', linestyle='--', alpha=0.5,
               label=f'Clean accuracy: {accs[0]:.1f}%')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    clean_patch = mpatches.Patch(color='#388E3C', label='Clean')
    perturb_patch = mpatches.Patch(color='#EF5350', label='Perturbed')
    ax.legend(handles=[clean_patch, perturb_patch], loc='lower right')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "robustness.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def error_analysis(model):
    """Show misclassified images from Config 3 best model."""
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"), transform=eval_transform
    )
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    raw_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"))
    all_paths    = [s[0] for s in raw_dataset.samples]

    misclassified = []
    with torch.no_grad():
        for idx, (img, label) in enumerate(loader):
            img = img.to(DEVICE)
            output = model(img)
            pred   = output.argmax(dim=1).item()
            if pred != label.item():
                misclassified.append({
                    "path":  all_paths[idx],
                    "true":  CLASSES[label.item()],
                    "pred":  CLASSES[pred],
                    "idx":   idx
                })

    print(f"\n  Misclassified: {len(misclassified)} / {len(test_dataset)} images")

    n = min(6, len(misclassified))
    if n == 0:
        print("  No misclassifications to show!")
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 8)) if n > 3 else plt.subplots(1, n, figsize=(4*n, 4))
    axes = np.array(axes).flatten()

    for i in range(n):
        m   = misclassified[i]
        img = Image.open(m["path"]).convert("RGB").resize((224, 224))
        axes[i].imshow(img)
        axes[i].set_title(f'True: {m["true"]}\nPred: {m["pred"]}',
                          color='red', fontsize=10, fontweight='bold')
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Error Analysis — Misclassified Images (Config 3 Best Model)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "error_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

if __name__ == "__main__":
    print("\n Running Analysis\n")

    # Load models
    print("Loading Config 1 (Baseline)")
    model1 = load_model(1)
    print("Loading Config 3 (Best)")
    model3 = load_model(3)

    test_data   = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    preds1, labels1, _ = get_predictions(model1, test_loader)
    preds3, labels3, _ = get_predictions(model3, test_loader)

    cm1 = make_cm(preds1, labels1)
    cm3 = make_cm(preds3, labels3)

    print("\n 1. Side-by-side confusion matrices")
    plot_side_by_side_cm(cm1, cm3)

    print("\n 2. Summary bar chart")
    plot_summary_bar()

    print("\n 3. Robustness testing (Config 3 best model)")
    robustness_results = robustness_test(model3)
    plot_robustness(robustness_results)

    print("\n 4. Error analysis")
    error_analysis(model3)

    print("\n Analysis complete! Check results/ folder for all plots.")
    print("\nFiles generated:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"   results/{f}")
