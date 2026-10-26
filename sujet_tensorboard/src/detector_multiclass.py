"""Option B : détecteur multi-classes (cercle, carré, triangle de couleurs variées).

Même principe que detector.py (TP), avec en plus la prédiction de la classe :
  Image 64x64 -> backbone (3 étages conv stride 2) -> feature map 32x8x8 (stride 8)
  Une conv 1x1 prédit pour chaque cellule 5 + C valeurs :
    - canal 0        : objectness (logit)
    - canaux 1, 2    : dx, dy in [0,1] (décalage du centre dans la cellule, après sigmoïde)
    - canaux 3, 4    : w, h in [0,1]  (taille relative à l'image, après sigmoïde)
    - canaux 5..5+C  : logits des C classes (softmax -> probabilités)

La cible d'entraînement est un tenseur (6, G, G) :
    (objectness, dx, dy, w, h, classe) ; la classe est un entier 0..C-1 stocké en float.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from src.detector import STRIDE, iou, nms

CLASSES = ("circle", "square", "triangle")   # même ordre que dataset_generator_2.py
NUM_CLASSES = len(CLASSES)


def code(cx: float, cy: float, w: float, h: float, class_id: int, size: int = 64) -> torch.Tensor:
    """Boîte [cx, cy, w, h] en pixels + classe -> cible (6, G, G) pour l'entraînement."""
    grid = size // STRIDE                       # G = 64 // 8 = 8 cellules par côté
    gx, gy = cx / size * grid, cy / size * grid  # centre exprimé en unités de cellules
    col, row = min(int(gx), grid - 1), min(int(gy), grid - 1)  # cellule responsable
    target = torch.zeros(6, grid, grid)         # toutes les cellules sont négatives...
    target[:, row, col] = torch.tensor(          # ... sauf celle qui contient le centre
        [1.0, gx - col, gy - row, w / size, h / size, float(class_id)])
    return target


def decode(logits: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(5+C, G, G) -> boîtes (G*G, 4) [x1, y1, x2, y2] en pixels, scores (G*G,), probas (G*G, C)."""
    grid = logits.shape[-1]
    p = torch.sigmoid(logits[:5])                       # objectness et boîte dans [0, 1]
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    cx = (cols + p[1]) * STRIDE                         # retour en pixels
    cy = (rows + p[2]) * STRIDE
    w, h = p[3] * size, p[4] * size
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    class_probs = torch.softmax(logits[5:], dim=0)      # (C, G, G), somme = 1 sur les classes
    return boxes.reshape(-1, 4), p[0].reshape(-1), class_probs.reshape(NUM_CLASSES, -1).T


def read_label(path: Path) -> tuple[float, float, float, float, int]:
    """Lit "cx cy w h class_id" (format de dataset_generator_2.py)."""
    cx, cy, w, h, class_id = path.read_text().split()
    return float(cx), float(cy), float(w), float(h), int(class_id)


class MultiShapeDataset(Dataset):
    """Images sur disque + étiquettes "cx cy w h class_id" -> (image (3,H,W), cible (6,G,G))."""

    def __init__(self, folder: Path) -> None:
        self.paths = sorted(Path(folder).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"Aucune image dans {folder} (lancer dataset_generator_2).")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[i]
        image = torch.from_numpy(
            np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        cx, cy, w, h, class_id = read_label(path.with_suffix(".txt"))
        return image, code(cx, cy, w, h, class_id, size=image.shape[-1])


def conv_block(c_in: int, c_out: int, stride: int) -> list[nn.Module]:
    """Conv 3x3 + normalisation par batch + ReLU (exigence de la question 2 du TP)."""
    return [nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(c_out), nn.ReLU()]


class MultiClassDetector(nn.Module):
    """Backbone 64x64 -> 32x8x8, puis tête conv 1x1 -> (5 + C) x 8 x 8.

    deep=False : backbone identique au TP (3 conv stride 2 : 3 -> 8 -> 16 -> 32 canaux).
    deep=True  : chaque étage contient en plus une convolution de stride 1 ; la sortie
    reste 32x8x8, mais le champ réceptif est plus grand et le réseau a plus de capacité
    (meilleure localisation de figures de formes et couleurs variées).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, deep: bool = True) -> None:
        super().__init__()
        if deep:
            layers = (conv_block(3, 16, stride=2) + conv_block(16, 16, stride=1)      # 64 -> 32
                      + conv_block(16, 32, stride=2) + conv_block(32, 32, stride=1)   # 32 -> 16
                      + conv_block(32, 32, stride=2) + conv_block(32, 32, stride=1))  # 16 -> 8
        else:
            layers = (conv_block(3, 8, stride=2) + conv_block(8, 16, stride=2)
                      + conv_block(16, 32, stride=2))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(32, 5 + num_classes, kernel_size=1)

    def feature_maps(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature_maps(images))


def compute_loss(logits: torch.Tensor, target: torch.Tensor,
                 box_weight: float = 10.0, class_weight: float = 1.0) -> torch.Tensor:
    """BCE objectness (toutes les cellules) + MSE boîte et entropie croisée classe (cellule positive)."""
    obj_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], target[:, 0], pos_weight=torch.tensor(10.0))
    positive = target[:, 0] > 0.5                                       # (B, G, G)
    pred_box = torch.sigmoid(logits[:, 1:5]).permute(0, 2, 3, 1)[positive]   # (B, 4)
    true_box = target[:, 1:5].permute(0, 2, 3, 1)[positive]
    box_loss = nn.functional.mse_loss(pred_box, true_box)
    class_logits = logits[:, 5:].permute(0, 2, 3, 1)[positive]          # (B, C)
    true_class = target[:, 5][positive].long()                          # (B,)
    class_loss = nn.functional.cross_entropy(class_logits, true_class)
    return obj_loss + box_weight * box_loss + class_weight * class_loss


def predict(logits: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, float, int, float]:
    """Une figure par image : boîte de plus fort score après NMS, avec sa classe."""
    boxes, scores, class_probs = decode(logits, size)
    best = nms(boxes, scores, iou_threshold=0.5)[0]
    class_prob, class_id = class_probs[best].max(dim=0)
    return boxes[best], float(scores[best]), int(class_id), float(class_prob)

