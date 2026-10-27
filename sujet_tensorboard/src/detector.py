"""Détecteur minimal : feature maps -> prédiction par cellule -> NMS -> une boîte.

Image 64x64 -> 3 blocs conv stride 2 -> feature map 32x8x8 (stride 8).
Une conv 1x1 prédit pour chaque cellule 5 valeurs : (objectness, dx, dy, w, h)
  - dx, dy in [0,1] : décalage du centre dans la cellule
  - w, h in [0,1]   : taille relative à l'image
"""

# --- Resumen en español ---
# Detector del TP (una sola clase). La idea general:
#   1. La imagen 64x64 pasa por un "backbone" convolucional que la reduce a una cuadrícula 8x8
#      (cada casilla o "celda" de la cuadrícula ve un trozo de 8x8 píxeles de la imagen).
#   2. Una "cabeza" (conv 1x1) predice 5 números por celda:
#        objectness = ¿hay un objeto cuyo centro cae en esta celda?
#        dx, dy     = dónde está el centro dentro de la celda (0 = borde izq./arriba, 1 = borde der./abajo)
#        w, h       = tamaño de la caja como fracción del tamaño de la imagen
#   3. Se decodifican las 64 celdas a cajas en píxeles y nos quedamos con la de mayor puntuación.

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

STRIDE = 8  # factor de reducción: 64 px de imagen -> 8 celdas (cada celda = 8 px)

def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU entre une boîte a (4,) et des boîtes b (N, 4)."""
    # IoU = área de la intersección / área de la unión. Vale 1 si las cajas coinciden, 0 si no se tocan.
    # Ancho y alto de la intersección; clamp(min=0) pone 0 cuando las cajas no se solapan
    inter_w = (torch.minimum(a[2], b[:, 2]) - torch.maximum(a[0], b[:, 0])).clamp(min=0)
    inter_h = (torch.minimum(a[3], b[:, 3]) - torch.maximum(a[1], b[:, 1])).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    # unión = área_a + área_b - intersección (si no, la zona común se contaría dos veces);
    # el clamp evita dividir por 0 con cajas de área nula
    return inter / (area_a + area_b - inter).clamp(min=1e-8)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 1.1) -> list[int]:
    """Garde la meilleure boîte, supprime celles qui la chevauchent, recommence et retourne les indices des boîtes conservées."""
    # Non-Maximum Suppression: entre varias cajas que se solapan, nos quedamos solo con la más segura.
    # Con el umbral por defecto 1.1 nunca se elimina nada (el IoU es <= 1): solo ordena por puntuación.
    order = scores.argsort(descending=True)  # índices de la mejor puntuación a la peor
    kept = []
    while len(order):
        best = order[0]           # la mejor caja que queda
        kept.append(int(best))    # se conserva
        order = order[1:]
        # se eliminan las cajas que se solapan demasiado con ella (IoU >= umbral)
        order = order[iou(boxes[best], boxes[order]) < iou_threshold]
    return kept


def code(cx: float, cy: float, w: float, h: float, size: int = 64) -> torch.Tensor:
    """Boîte [cx, cy, w, h] en pixels -> (5, G, G) pour l'entraînement."""
    # Convierte la etiqueta en píxeles en el "objetivo" que la red debe aprender a producir.
    grid = size // STRIDE                         # G = 8 celdas por lado
    gx, gy = cx / size * grid, cy / size * grid   # centro expresado en unidades de celda (ej. 4.7)
    # Celda responsable = la que contiene el centro (parte entera); min() por si el centro cae justo en el borde
    col, row = min(int(gx), grid - 1), min(int(gy), grid - 1)
    target = torch.zeros(5, grid, grid)           # todas las celdas a 0 = "aquí no hay objeto"...
    # ...salvo la celda responsable: objectness=1, posición dentro de la celda (parte decimal), tamaño relativo
    target[:, row, col] = torch.tensor([1.0, gx - col, gy - row, w / size, h / size])
    return target

def decode(logits: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """(5, G, G) -> boîtes (G*G, 4) en pixels [x1, y1, x2, y2] et scores (G*G,)."""
    # Operación inversa de code(): pasa de la salida de la red a cajas en píxeles.
    grid = logits.shape[-1]
    p = torch.sigmoid(logits)  # la red da números cualesquiera; sigmoid los lleva a [0, 1]
    # rows[i, j] = i y cols[i, j] = j: número de fila/columna de cada celda
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    cx = (cols + p[1]) * STRIDE   # (número de celda + posición dentro de la celda) * 8 px
    cy = (rows + p[2]) * STRIDE
    w, h = p[3] * size, p[4] * size  # fracción de la imagen -> píxeles
    # Paso de (centro, tamaño) a esquinas [x1, y1, x2, y2]
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    # Se aplanan las 8x8 celdas en una lista de 64 cajas y 64 puntuaciones
    return boxes.reshape(-1, 4), p[0].reshape(-1)



class ShapeDataset(Dataset):
    """Cible (5, G, G) : tout à 0 sauf la cellule contenant le centre."""
    # Lee las imágenes y etiquetas guardadas en disco por dataset_generator.py

    def __init__(self, folder: Path, size: int = 64) -> None:
        self.paths = sorted(Path(folder).glob("*.png"))  # lista de todas las imágenes
        if not self.paths:
            raise FileNotFoundError(f"Aucune image dans {folder} (lancer dataset_generator).")
        self.grid = size // STRIDE

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Devuelve el par (imagen, objetivo) número i
        path = self.paths[i]
        # PNG -> array (H, W, 3) -> tensor (3, H, W) con valores en [0, 1] (formato que espera Conv2d)
        image = torch.from_numpy(
            np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        cx, cy, w, h = map(float, path.with_suffix(".txt").read_text().split())  # lee "cx cy w h" del .txt
        target = code(cx, cy, w, h, size=image.shape[-1])  # etiqueta -> objetivo (5, 8, 8)
        return image, target


class Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Backbone: 3 bloques "convolución 3x3 con stride 2 + BatchNorm + ReLU".
        # Cada bloque divide el tamaño por 2 (64 -> 32 -> 16 -> 8) y aumenta los canales (3 -> 8 -> 16 -> 32).
        blocks = [nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                  nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        self.backbone = nn.Sequential(*blocks)
        # Cabeza: conv 1x1 que transforma los 32 valores de cada celda en las 5 predicciones
        self.head = nn.Conv2d(32, 5, kernel_size=1, stride = 1)


    def feature_maps(self, images: torch.Tensor) -> torch.Tensor:
        # (B, 3, 64, 64) -> (B, 32, 8, 8): 32 "mapas de características" de 8x8
        feature_maps = self.backbone(images)

        return feature_maps

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (B, 3, 64, 64) -> (B, 5, 8, 8): las 5 predicciones de cada celda (antes de sigmoid)
        feature_maps = self.feature_maps(images)
        output = self.head(feature_maps)

        return output





def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE sur l'objectness de toutes les cellules + MSE sur le décalage/taille de la cellule positive."""
    # 1) ¿Hay objeto? Entropía cruzada binaria en TODAS las celdas.
    #    pos_weight=10 da más peso a la única celda positiva (hay 1 positiva por 63 negativas).
    obj_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], target[:, 0], pos_weight=torch.tensor(10.0))
    # 2) ¿Dónde y de qué tamaño? Error cuadrático medio, solo en la celda que contiene el objeto
    positive = target[:, 0] > 0.5  # (B, G, G) máscara True en la celda positiva de cada imagen
    # permute -> (B, G, G, 4) para poder seleccionar con la máscara -> (B, 4)
    pred_box = torch.sigmoid(logits[:, 1:]).permute(0, 2, 3, 1)[positive]
    true_box = target[:, 1:].permute(0, 2, 3, 1)[positive]
    # El factor 10 equilibra los dos términos (la MSE de valores en [0, 1] es muy pequeña)
    return obj_loss + 10.0 * nn.functional.mse_loss(pred_box, true_box)




def predict(logits: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, float]:
    """Une seule figure par image : la boîte de plus fort score après NMS."""
    # Salida de la red para UNA imagen (5, 8, 8) -> caja final en píxeles + su puntuación.
    # Se usa en la evaluación y para dibujar resultados (no en el entrenamiento).
    boxes, scores = decode(logits, size)
    best = nms(boxes, scores)[0]  # el primer índice devuelto es la caja de mayor puntuación
    return boxes[best], float(scores[best])
