"""Option B : entraîne le détecteur multi-classes et l'évalue (localisation + classification).

Utilisation (depuis le dossier sujet_tensorboard) :
    python -m src.train_multiclass --epochs 40
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.dataset_generator_2 import generate
from src.detector_multiclass import (CLASSES, NUM_CLASSES, MultiClassDetector, MultiShapeDataset,
                                     compute_loss, iou, predict)


def train(model: MultiClassDetector, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Entraîne le modèle sur une époque et retourne la perte moyenne par batch."""
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        optimizer.zero_grad()
        loss = compute_loss(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def target_to_label(target: torch.Tensor, size: int) -> tuple[torch.Tensor, int]:
    """Cible (6, G, G) -> boîte vraie [x1, y1, x2, y2] en pixels et classe vraie."""
    grid = target.shape[-1]
    row, col = divmod(int(target[0].argmax()), grid)       # la seule cellule positive
    _, dx, dy, w, h, class_id = target[:, row, col].tolist()
    cx, cy = (col + dx) * size / grid, (row + dy) * size / grid
    w, h = w * size, h * size
    return torch.tensor([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]), int(class_id)


def evaluate(model: MultiClassDetector, loader: DataLoader, size: int) -> dict:
    """IoU moyen, % IoU > 0.5, précision de classification, matrice de confusion, IoU par classe."""
    model.eval()
    ious, true_classes, pred_classes = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            for logit, target in zip(model(images), targets):
                box, _, class_id, _ = predict(logit, size)
                truth, true_class = target_to_label(target, size)
                ious.append(float(iou(box, truth[None])[0]))
                true_classes.append(true_class)
                pred_classes.append(class_id)
    ious = torch.tensor(ious)
    true_classes = torch.tensor(true_classes)
    pred_classes = torch.tensor(pred_classes)
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)  # lignes = vraie classe
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1
    good_class = pred_classes == true_classes
    return {
        "mean_iou": ious.mean().item(),
        "iou_gt_05": (ious > 0.5).float().mean().item(),
        "accuracy": good_class.float().mean().item(),
        # détection "complète" correcte : bonne boîte (IoU > 0.5) ET bonne classe
        "detection_ok": ((ious > 0.5) & good_class).float().mean().item(),
        "confusion": confusion,
        "iou_per_class": [ious[true_classes == c].mean().item() for c in range(NUM_CLASSES)],
        "ious": ious,
    }


def plot_history(history: dict, path: Path | None = None) -> None:
    """Courbes d'apprentissage : perte d'entraînement et métriques de validation."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(epochs, history["loss"])
    axes[0].set_title("Perte d'entraînement")
    axes[0].set_xlabel("époque")
    for key, label in (("mean_iou", "IoU moyen"), ("iou_gt_05", "IoU > 0,5"),
                       ("accuracy", "précision classe"), ("detection_ok", "IoU > 0,5 et bonne classe")):
        axes[1].plot(epochs, history[key], label=label)
    axes[1].set_title("Validation")
    axes[1].set_xlabel("époque")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data_multiclass"))
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--val", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    if not (args.data / "train").exists():
        generate(args.data, size=args.size, train=args.train, val=args.val)
    train_loader = DataLoader(MultiShapeDataset(args.data / "train"), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MultiShapeDataset(args.data / "val"), batch_size=args.batch_size)

    model = MultiClassDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {k: [] for k in ("loss", "mean_iou", "iou_gt_05", "accuracy", "detection_ok")}
    for epoch in range(1, args.epochs + 1):
        history["loss"].append(train(model, train_loader, optimizer))
        metrics = evaluate(model, val_loader, args.size)
        for key in ("mean_iou", "iou_gt_05", "accuracy", "detection_ok"):
            history[key].append(metrics[key])
        print(f"epoch {epoch:2d}  loss {history['loss'][-1]:.4f}  IoU {metrics['mean_iou']:.3f}  "
              f"IoU>0.5 {metrics['iou_gt_05']:.0%}  acc {metrics['accuracy']:.0%}")

    print("Matrice de confusion (lignes = vraie classe, colonnes = prédite) :", CLASSES)
    print(metrics["confusion"])
    torch.save(model.state_dict(), "detector_multiclass.pt")
    plot_history(history, Path("courbes_multiclass.png"))


if __name__ == "__main__":
    main()
