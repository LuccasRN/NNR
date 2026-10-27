"""Générateur de données : une figure par image, de classe et couleur aléatoires.

Classes : circle=0, square=1, triangle=2.
Étiquette : "cx cy w h class_id" (coordonnées en pixels).
"""

# --- Resumen en español ---
# Generador de la OPCIÓN B. Igual que dataset_generator.py (una sola figura por imagen),
# pero ahora:
#   - la FORMA se elige al azar entre círculo (0), cuadrado (1) y triángulo (2);
#   - el COLOR se elige al azar entre 5 colores, independiente de la forma.
#     Así el color no da ninguna pista: la red está obligada a aprender la forma.
#   - la etiqueta tiene un quinto número: "cx cy w h class_id".

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle")  # el índice en esta tupla ES el número de clase
COLORS = (
    (230, 75, 60),    # rouge
    (50, 130, 220),   # bleu
    (50, 175, 100),   # vert
    (235, 180, 45),   # jaune
    (160, 80, 200),   # violet
)
OUTLINE = (25, 25, 25)  # borde casi negro


def draw_shape(draw, shape, box, color):
    # Igual que en el TP, pero el color de relleno se pasa como parámetro
    x1, y1, x2, y2 = box

    if shape == "circle":
        draw.ellipse(box, fill=color, outline=OUTLINE)
    elif shape == "square":
        draw.rectangle(box, fill=color, outline=OUTLINE)
    else:
        # Triángulo: vértice arriba al centro, base abajo
        draw.polygon(
            ((x1 + (x2 - x1) // 2, y1),
             (x2, y2),
             (x1, y2)),
            fill=color,
            outline=OUTLINE,
        )


def make_example(size, rng):
    # Fondo gris claro uniforme
    background = rng.randint(225, 250)
    image = Image.new("RGB", (size, size), (background,) * 3)

    # Une seule figure par image
    class_id = rng.randrange(len(SHAPES))  # clase al azar: 0, 1 o 2
    shape = SHAPES[class_id]               # nombre de la forma correspondiente
    color = rng.choice(COLORS)             # color al azar, sin relación con la forma

    # Tamaño y posición aleatorios (mismas reglas que en el TP)
    side = rng.randint(size // 6, size // 3)
    x1 = rng.randint(0, size - side - 1)
    y1 = rng.randint(0, size - side - 1)
    x2, y2 = x1 + side, y1 + side

    draw_shape(
        ImageDraw.Draw(image),
        shape,
        (x1, y1, x2, y2),
        color,
    )

    # Centro de la figura en píxeles
    cx = x1 + side / 2
    cy = y1 + side / 2

    # Etiqueta: centro, ancho, alto y número de clase
    label = f"{cx} {cy} {side} {side} {class_id}"
    return image, label


def generate(
    output: Path,
    size: int = 64,
    train: int = 400,
    val: int = 100,
    seed: int = 7,
):
    # Guarda las imágenes (.png) y sus etiquetas (.txt) en output/train y output/val
    if size % 8:
        raise ValueError("size doit être un multiple de 8.")   # por el stride 8 del detector
    if size < 24:
        raise ValueError("size doit être supérieur ou égal à 24.")  # si no, las figuras serían demasiado pequeñas

    rng = random.Random(seed)  # semilla fija -> datos reproducibles

    for split, total in (("train", train), ("val", val)):
        folder = output / split
        folder.mkdir(parents=True, exist_ok=True)

        for i in range(total):
            image, label = make_example(size, rng)
            image.save(folder / f"{i:05d}.png")
            (folder / f"{i:05d}.txt").write_text(
                label + "\n", encoding="utf-8"
            )

    print(
        f"{train + val} images générées dans {output.resolve()}"
    )
    print("Classes : circle=0, square=1, triangle=2")
