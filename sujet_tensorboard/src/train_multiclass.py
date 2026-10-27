"""Option B : entraîne le détecteur multi-classes et l'évalue (localisation + classification).

Utilisation (depuis le dossier sujet_tensorboard) :
    python -m src.train_multiclass --epochs 40
"""

# --- Resumen en español ---
# Entrenamiento y evaluación del detector multiclase (opción B).
#   train()          -> una época de aprendizaje
#   evaluate()       -> métricas de localización (IoU) y de clasificación (precisión, matriz de confusión)
#   plot_history()   -> curvas de aprendizaje con matplotlib
#   main()           -> script completo; también escribe las curvas en TensorBoard

import argparse
import random
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset_generator_2 import generate
from src.detector_multiclass import (CLASSES, NUM_CLASSES, MultiClassDetector, MultiShapeDataset,
                                     compute_loss, iou, predict)


def train(model: MultiClassDetector, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Entraîne le modèle sur une époque et retourne la perte moyenne par batch."""
    model.train()                                   # modo entrenamiento (BatchNorm activo)
    total_loss = 0.0
    for images, targets in loader:
        optimizer.zero_grad()                       # borra los gradientes anteriores
        loss = compute_loss(model(images), targets) # predicción + error
        loss.backward()                             # gradientes
        optimizer.step()                            # actualización de los pesos
        total_loss += loss.item()
    return total_loss / len(loader)                 # media por batch (aquí sí es la pérdida real)


def target_to_label(target: torch.Tensor, size: int) -> tuple[torch.Tensor, int]:
    """Cible (6, G, G) -> boîte vraie [x1, y1, x2, y2] en pixels et classe vraie."""
    grid = target.shape[-1]
    row, col = divmod(int(target[0].argmax()), grid)       # la única celda positiva
    _, dx, dy, w, h, class_id = target[:, row, col].tolist()
    cx, cy = (col + dx) * size / grid, (row + dy) * size / grid  # centro en píxeles
    w, h = w * size, h * size                                    # tamaño en píxeles
    return torch.tensor([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]), int(class_id)


def evaluate(model: MultiClassDetector, loader: DataLoader, size: int) -> dict:
    """IoU moyen, % IoU > 0.5, précision de classification, matrice de confusion, IoU par classe."""
    model.eval()
    ious, true_classes, pred_classes = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            for logit, target in zip(model(images), targets):
                box, _, class_id, _ = predict(logit, size)     # lo que dice la red
                truth, true_class = target_to_label(target, size)  # la respuesta correcta
                ious.append(float(iou(box, truth[None])[0]))
                true_classes.append(true_class)
                pred_classes.append(class_id)
    ious = torch.tensor(ious)
    true_classes = torch.tensor(true_classes)
    pred_classes = torch.tensor(pred_classes)
    # Matriz de confusión: confusion[verdadera, predicha] = número de imágenes en ese caso.
    # Una red perfecta solo tiene números en la diagonal.
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)  # lignes = vraie classe
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1
    good_class = pred_classes == true_classes
    return {
        "mean_iou": ious.mean().item(),                           # IoU medio
        "iou_gt_05": (ious > 0.5).float().mean().item(),          # % de cajas "buenas" (IoU > 0.5)
        "accuracy": good_class.float().mean().item(),             # % de clases correctas
        # détection "complète" correcte : bonne boîte (IoU > 0.5) ET bonne classe
        "detection_ok": ((ious > 0.5) & good_class).float().mean().item(),
        "confusion": confusion,
        "iou_per_class": [ious[true_classes == c].mean().item() for c in range(NUM_CLASSES)],
        "ious": ious,                                             # todos los IoU (para histogramas)
    }


def plot_history(history: dict, path: Path | None = None) -> None:
    """Courbes d'apprentissage : perte d'entraînement et métriques de validation."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    # Izquierda: pérdida de entrenamiento (debe bajar)
    axes[0].plot(epochs, history["loss"])
    axes[0].set_title("Perte d'entraînement")
    axes[0].set_xlabel("époque")
    # Derecha: métricas de validación (deben subir hacia 1)
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
    # Parámetros configurables desde la línea de comandos (ej. --epochs 20)
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
    if not (args.data / "train").exists():  # genera los datos en disco si aún no existen
        generate(args.data, size=args.size, train=args.train, val=args.val)
    train_loader = DataLoader(MultiShapeDataset(args.data / "train"), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MultiShapeDataset(args.data / "val"), batch_size=args.batch_size)

    model = MultiClassDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {k: [] for k in ("loss", "mean_iou", "iou_gt_05", "accuracy", "detection_ok")}
    # Logs de TensorBoard fuera de la carpeta sincronizada (mismo truco que demo.ipynb)
    run_dir = Path(tempfile.gettempdir()) / "tb_runs" / "option_B" / datetime.now().strftime("script-%Y%m%d-%H%M%S")
    print("Logs TensorBoard :", run_dir)
    with SummaryWriter(log_dir=str(run_dir)) as writer:
        for epoch in range(1, args.epochs + 1):
            history["loss"].append(train(model, train_loader, optimizer))
            metrics = evaluate(model, val_loader, args.size)
            for key in ("mean_iou", "iou_gt_05", "accuracy", "detection_ok"):
                history[key].append(metrics[key])
            # Una curva por métrica en TensorBoard, actualizada en cada época
            writer.add_scalar("Loss/train", history["loss"][-1], epoch)
            writer.add_scalar("Validation/IoU_moyen", metrics["mean_iou"], epoch)
            writer.add_scalar("Validation/IoU_gt_0.5", metrics["iou_gt_05"], epoch)
            writer.add_scalar("Validation/precision_classe", metrics["accuracy"], epoch)
            writer.add_scalar("Validation/detection_ok", metrics["detection_ok"], epoch)
            print(f"epoch {epoch:2d}  loss {history['loss'][-1]:.4f}  IoU {metrics['mean_iou']:.3f}  "
                  f"IoU>0.5 {metrics['iou_gt_05']:.0%}  acc {metrics['accuracy']:.0%}")

    print("Matrice de confusion (lignes = vraie classe, colonnes = prédite) :", CLASSES)
    print(metrics["confusion"])
    torch.save(model.state_dict(), "detector_multiclass.pt")
    plot_history(history, Path("courbes_multiclass.png"))


if __name__ == "__main__":
    main()
