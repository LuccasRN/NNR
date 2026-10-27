"""Entraîne le détecteur, évalue (IoU moyen, % IoU > 0.5) et sauvegarde une figure."""

# --- Resumen en español ---
# Bucle de entrenamiento (una época) y evaluación del detector del TP.

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.detector import Detector, ShapeDataset, compute_loss, iou, predict


def train(model: Detector, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Entraîne le modèle sur une époque et retourne la perte moyenne."""
    model.train()  # modo entrenamiento: BatchNorm usa las estadísticas del batch actual
    total_loss = 0.0
    for images, targets in loader:  # el loader entrega los datos por lotes (batches)
        optimizer.zero_grad()                  # borra los gradientes del paso anterior
        outputs = model(images)                # predicciones de la red
        loss = compute_loss(outputs, targets)  # cuánto se equivoca la red
        loss.backward()                        # calcula los gradientes (en qué dirección corregir cada peso)
        optimizer.step()                       # corrige los pesos un poco en esa dirección
        total_loss += loss.item()              # acumula para hacer la media al final
    # Ojo: loss ya es una media por batch y aquí se divide por el número de IMÁGENES, no de batches,
    # así que el valor devuelto es ~64 veces más pequeño que la pérdida real (solo cambia la escala).
    return total_loss / len(loader.dataset)


def evaluate(model: Detector, loader: DataLoader, size: int) -> tuple[float, float]:
    # Devuelve (IoU medio, proporción de imágenes con IoU > 0.5) sobre el loader
    model.eval()  # modo evaluación: BatchNorm usa las estadísticas aprendidas
    ious = []
    with torch.no_grad():  # no hacen falta gradientes -> más rápido
        for images, targets in loader:
            logits = model(images)
            grid = targets.shape[-1]
            for logit, target in zip(logits, targets):
                box, _ = predict(logit, size)  # caja predicha en píxeles
                # Reconstrucción de la caja verdadera a partir del objetivo (inverso de code)
                row, col = divmod(int(target[0].argmax()), grid)  # celda positiva
                _, dx, dy, w, h = target[:, row, col].tolist()
                cx, cy = (col + dx) * size / grid, (row + dy) * size / grid
                truth = torch.tensor([cx - w * size / 2, cy - h * size / 2,
                                      cx + w * size / 2, cy + h * size / 2])
                ious.append(float(iou(box, truth[None])[0]))  # truth[None]: (4,) -> (1, 4)
    t = torch.tensor(ious)
    return t.mean().item(), (t > 0.5).float().mean().item()
