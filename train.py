"""
train.py
--------
Plant Leaf Disease Classifier - MSML640 Final Project
Author: Hari Haran Manda

Trains EfficientNet-B0 under 4 configurations:
  1. Baseline          - original data, no augmentation
  2. Augmented         - original data + augmentation
  3. Synthesized       - original + synthetic images
  4. Full Pipeline     - original + synthetic + augmentation

Usage:
  python train.py --config 1   # Run baseline
  python train.py --config 2   # Run augmented
  python train.py --config 3   # Run synthesized
  python train.py --config 4   # Run full pipeline
  python train.py --config all # Run all 4 configs
"""

import os
import argparse
import copy
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = "data"          
SYNTH_DIR  = "synth_data"    
OUTPUT_DIR = "results"       
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES      = ["healthy", "early_blight", "late_blight"]
NUM_CLASSES  = len(CLASSES)
BATCH_SIZE   = 16
NUM_EPOCHS   = 15
LR           = 1e-4         
IMG_SIZE     = 224            
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

aug_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  
])

eval_transform = base_transform


def build_datasets(config_id):
    """
    Returns train_dataset, val_dataset, test_dataset for a given config.
    Config 1: baseline (original train data, no aug)
    Config 2: augmented (original train data, with aug)
    Config 3: synthesized (original + synthetic train data, no aug)
    Config 4: full pipeline (original + synthetic train data, with aug)
    """
    use_aug   = config_id in [2, 4]
    use_synth = config_id in [3, 4]

    train_transform = aug_transform if use_aug else base_transform

    train_data = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "train"),
        transform=train_transform
    )

    if use_synth and os.path.exists(SYNTH_DIR):
        synth_data = datasets.ImageFolder(
            root=os.path.join(SYNTH_DIR, "train"),
            transform=train_transform
        )
        train_data = torch.utils.data.ConcatDataset([train_data, synth_data])
        print(f"  [Config {config_id}] Added synthetic data. Total train samples: {len(train_data)}")
    elif use_synth:
        print(f"  [Config {config_id}] synth_data/ folder not found — running without synthetic images.")

    val_data  = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),  transform=eval_transform)
    test_data = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_transform)

    return train_data, val_data, test_data


def build_model():
    """
    Load EfficientNet-B0 pretrained on ImageNet.
    Replace the final classifier for our 3-class task.
    Freeze all layers except the final classifier for fast fine-tuning.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES)
    )

    return model.to(DEVICE)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct  += preds.eq(labels).sum().item()
            total    += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def plot_curves(train_losses, val_losses, train_accs, val_accs, config_id):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, val_losses,   'r-o', label='Val Loss',   markersize=4)
    ax1.set_title(f'Config {config_id} — Loss Curves')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a*100 for a in train_accs], 'b-o', label='Train Acc', markersize=4)
    ax2.plot(epochs, [a*100 for a in val_accs],   'r-o', label='Val Acc',   markersize=4)
    ax2.set_title(f'Config {config_id} — Accuracy Curves')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 100); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"config{config_id}_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(preds, labels, config_id):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Greens')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=30, ha='right')
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Config {config_id} — Confusion Matrix')
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"config{config_id}_confusion.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")
    return cm

def run_config(config_id):
    config_names = {
        1: "Baseline (no aug, no synth)",
        2: "Augmented",
        3: "Synthesized data",
        4: "Full Pipeline (synth + aug)"
    }
    print(f"\n{'='*60}")
    print(f"  CONFIG {config_id}: {config_names[config_id]}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    train_data, val_data, test_data = build_datasets(config_id)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc  = 0.0
    best_weights  = None

    for epoch in range(1, NUM_EPOCHS + 1):
        t_loss, t_acc          = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, _, _    = evaluate(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc);   val_accs.append(v_acc)

        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS}  "
              f"Train Loss: {t_loss:.4f}  Train Acc: {t_acc*100:.1f}%  "
              f"Val Loss: {v_loss:.4f}  Val Acc: {v_acc*100:.1f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)

    print(f"\n  Test Accuracy: {test_acc*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(labels, preds, target_names=CLASSES))

    plot_curves(train_losses, val_losses, train_accs, val_accs, config_id)
    plot_confusion_matrix(preds, labels, config_id)

    model_path = os.path.join(OUTPUT_DIR, f"config{config_id}_model.pth")
    torch.save(best_weights, model_path)
    print(f"  Model saved: {model_path}")

    return {
        "config":    config_id,
        "test_acc":  test_acc,
        "test_loss": test_loss,
        "preds":     preds,
        "labels":    labels
    }

def plot_summary(results):
    configs  = [f"Config {r['config']}" for r in results]
    accs     = [r['test_acc'] * 100 for r in results]
    colors   = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(configs, accs, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Comparison: All 4 Configurations')
    ax.axhline(y=max(accs), color='red', linestyle='--', alpha=0.5, label=f'Best: {max(accs):.1f}%')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    labels_legend = ['Baseline', 'Augmented', 'Synthesized', 'Full Pipeline']
    patches = [mpatches.Patch(color=colors[i], label=labels_legend[i]) for i in range(4)]
    ax.legend(handles=patches, loc='lower right')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "summary_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"\n  Summary chart saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="all",
                        help="Which config to run: 1, 2, 3, 4, or 'all'")
    args = parser.parse_args()

    if args.config == "all":
        configs_to_run = [1, 2, 3, 4]
    else:
        configs_to_run = [int(args.config)]

    results = []
    for cfg in configs_to_run:
        result = run_config(cfg)
        results.append(result)

    if len(results) == 4:
        plot_summary(results)
        print("\n All 4 configs complete! Check the results/ folder for all plots.")
    else:
        r = results[0]
        print(f"\n Config {r['config']} complete! Test Accuracy: {r['test_acc']*100:.2f}%")
