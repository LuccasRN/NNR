"""Simulateur : une seule figure (toujours la même) par image, position/taille aléatoires.

Étiquette (un fichier .txt par image) : "cx cy w h" normalisés dans [0, 1].
"""

# --- Resumen en español ---
# Este es el generador de datos del TP: crea imágenes sintéticas de 64x64 con UNA sola figura
# (siempre la misma forma y el mismo color rojo) en una posición y tamaño aleatorios.
# Para cada imagen devuelve también su etiqueta "cx cy w h":
#   cx, cy = centro de la figura (en píxeles)
#   w, h   = ancho y alto de la figura (en píxeles; aquí siempre w == h porque es un cuadrado de lado 'side')
# Ojo: el docstring dice "normalizados en [0, 1]" pero en realidad el código las escribe EN PÍXELES.

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle")  # formas que se saben dibujar
COLOR = (230, 75, 60)                      # color de relleno (rojo), siempre el mismo en el TP
OUTLINE = (25, 25, 25)                     # borde casi negro alrededor de la figura


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: tuple[int, int, int, int]) -> None:
    # Dibuja la figura dentro de la caja box = (x1, y1, x2, y2) (esquina sup. izquierda e inf. derecha)
    x1, y1, x2, y2 = box
    if shape == "circle":
        draw.ellipse(box, fill=COLOR, outline=OUTLINE)    # círculo inscrito en la caja
    elif shape == "square":
        draw.rectangle(box, fill=COLOR, outline=OUTLINE)  # el cuadrado ocupa toda la caja
    else:
        # Triángulo: vértice arriba en el medio, y los dos vértices de abajo en las esquinas
        draw.polygon(((x1 + (x2 - x1) // 2, y1), (x2, y2), (x1, y2)), fill=COLOR, outline=OUTLINE)


def make_example(size: int, shape: str, rng: random.Random) -> tuple[Image.Image, str]:
    # Fondo gris claro uniforme (un poco distinto en cada imagen)
    background = rng.randint(225, 250)
    image = Image.new("RGB", (size, size), (background,) * 3)
    # Lado de la figura: entre 1/6 y 1/3 del tamaño de la imagen (de 10 a 21 px para 64x64)
    side = rng.randint(size // 6, size // 3)
    # Esquina superior izquierda aleatoria, de modo que la figura quede entera dentro de la imagen
    x1 = rng.randint(0, size - side - 1)
    y1 = rng.randint(0, size - side - 1)
    draw_shape(ImageDraw.Draw(image), shape, (x1, y1, x1 + side, y1 + side))
    # Etiqueta en texto: "cx cy w h" (centro y tamaño en píxeles)
    label = (f"{(x1 + side / 2)} {(y1 + side / 2)} "
             f"{side} {side}")
    return image, label


def generate(output: Path, size: int = 64, train: int = 400, val: int = 100,
             shape: str = "circle", seed: int = 7) -> None:
    # Guarda en disco un conjunto de entrenamiento y uno de validación:
    # output/train/00000.png + 00000.txt, ..., output/val/...
    if size % 8:
        # El detector reduce la imagen 8 veces (stride 8): el tamaño debe ser múltiplo de 8
        raise ValueError("size doit être un multiple de 8.")
    rng = random.Random(seed)  # semilla fija -> siempre las mismas imágenes
    for split, total in (("train", train), ("val", val)):
        (output / split).mkdir(parents=True, exist_ok=True)
        for i in range(total):
            image, label = make_example(size, shape, rng)
            image.save(output / split / f"{i:05d}.png")
            (output / split / f"{i:05d}.txt").write_text(label + "\n", encoding="utf-8")
    print(f"{train + val} images ({shape}) générées dans {output.resolve()}")

