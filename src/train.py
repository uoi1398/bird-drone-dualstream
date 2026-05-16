#%%writefile src/train.py
import os
import json
import yaml
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score

from dataset import BirdDroneVideoDataset
from model import FrameAverageResNet


def set_seed(seed=42):
    """
    固定随机种子，保证实验尽量可复现。
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def compute_metrics(labels, preds):
    """
    返回 accuracy、macro-F1 和 drone 类 F1。
    macro-F1 更适合你这个 bird/drone 不平衡的小样本数据。
    """
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    drone_f1 = f1_score(labels, preds, pos_label=1, zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "drone_f1": drone_f1
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(dataloader, desc="Train", leave=False)

    for x, y, video_id in progress_bar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)

        all_labels.extend(y.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(dataloader, desc="Val", leave=False)

    for x, y, video_id in progress_bar:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)

        all_labels.extend(y.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


def build_dataloader(cfg, split):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    dataset = BirdDroneVideoDataset(
        csv_path=data_cfg["manifest_path"],
        split=split,
        mode=data_cfg["mode"],
        num_frames=model_cfg["num_frames"],
        image_size=model_cfg["image_size"],
        degrade_level=data_cfg.get("degrade_level", "none")
    )

    shuffle = True if split == "train" else False

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=True
    )

    return dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml file, e.g. configs/rgb.yaml"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    output_cfg = cfg["output"]
    run_dir = output_cfg["run_dir"]
    best_model_path = output_cfg["best_model_path"]

    ensure_dir(run_dir)
    ensure_dir(os.path.dirname(best_model_path))

    # 保存一份 config 到 run 目录，方便以后复查。
    with open(os.path.join(run_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    model = FrameAverageResNet(
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_name = train_cfg.get("optimizer", "AdamW")

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 0.0)
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    epochs = train_cfg["epochs"]

    best_val_macro_f1 = -1.0
    best_epoch = -1

    log_records = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}]")

        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_drone_f1": train_metrics["drone_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_drone_f1": val_metrics["drone_f1"]
        }

        log_records.append(record)

        print(
            f"Train loss={train_loss:.4f}, "
            f"acc={train_metrics['accuracy']:.4f}, "
            f"macro-F1={train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val   loss={val_loss:.4f}, "
            f"acc={val_metrics['accuracy']:.4f}, "
            f"macro-F1={val_metrics['macro_f1']:.4f}"
        )

        # 以 validation macro-F1 保存最佳模型
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch

            torch.save(model.state_dict(), best_model_path)

            print(
                f"Saved best model to {best_model_path} "
                f"(val_macro_f1={best_val_macro_f1:.4f})"
            )

        # 每轮都更新训练日志
        log_df = pd.DataFrame(log_records)
        log_df.to_csv(os.path.join(run_dir, "train_log.csv"), index=False)

    # 保存最后一轮模型
    last_model_path = os.path.join(run_dir, "last.pth")
    torch.save(model.state_dict(), last_model_path)

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_model_path": best_model_path,
        "last_model_path": last_model_path
    }

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro-F1: {best_val_macro_f1:.4f}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()