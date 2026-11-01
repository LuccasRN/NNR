"""Genera las ilustraciones de guia_detallada.md con la red YA entrenada (detector_multiclass.pt).

Uso (desde la carpeta sujet_tensorboard):
    python notebooks/guia_img/hacer_figuras.py
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # carpeta sujet_tensorboard
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, Rectangle

from src.dataset_generator_2 import COLORS, make_example
from src.detector_multiclass import MultiClassDetector, code, decode, predict
from src.detector import iou

OUT = Path(__file__).parent
NOMBRES = ("círculo", "cuadrado", "triángulo")
COLOR_CLASE = ("tab:blue", "tab:orange", "tab:green")
torch.set_grad_enabled(False)


def to_tensor(image):
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy()).permute(2, 0, 1) / 255.0


def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("ok", name)


def caja(ax, x1, y1, x2, y2, color, lw=2, ls="-", **kw):
    # imshow pone el centro del píxel i en i: las coordenadas en píxeles "continuas" se desplazan -0.5
    ax.add_patch(Rectangle((x1 - .5, y1 - .5), x2 - x1, y2 - y1, fill=False, ec=color, lw=lw, ls=ls, **kw))


def rejilla(ax, n=8, paso=8, color="white"):
    for k in range(n + 1):
        ax.axhline(k * paso - .5, color=color, lw=.6, alpha=.7)
        ax.axvline(k * paso - .5, color=color, lw=.6, alpha=.7)


def tabla(ax, arr, fmt="{:g}", cmap="Greys_r", vmin=None, vmax=None, fs=10, resaltar=None, color_res="red"):
    """Dibuja una matriz como cuadrícula con su número en cada casilla."""
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    for (r, c), v in np.ndenumerate(arr):
        if np.isnan(v):
            continue
        R, G, B, _ = im.cmap(im.norm(v))                  # texto blanco sobre fondo oscuro y al revés
        ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=fs,
                color="white" if .3 * R + .6 * G + .1 * B < .5 else "black")
    ax.set_xticks(np.arange(-.5, arr.shape[1]), [])
    ax.set_yticks(np.arange(-.5, arr.shape[0]), [])
    ax.grid(color="gray", lw=.5)
    ax.tick_params(length=0)
    if resaltar:
        r, c, h, w = resaltar
        ax.add_patch(Rectangle((c - .5, r - .5), w, h, fill=False, ec=color_res, lw=3))


# ---------- datos y modelo (los mismos que el notebook) ----------
rng = random.Random(123)
examples = [make_example(64, rng) for _ in range(10)]
EJ = next(i for i, (_, lb) in enumerate(examples) if lb.startswith("25.0 14.0 16 16 1"))  # el ejemplo de la guía
img_pil, label = examples[EJ]
img = to_tensor(img_pil)
cx, cy, w, h, cls = (float(v) for v in label.split())
img_np = np.asarray(img_pil)

model = MultiClassDetector(deep=True)
model.load_state_dict(torch.load(ROOT / "detector_multiclass.pt"))
model.eval()

rng = random.Random(8)                                    # mismo conjunto de validación que el notebook
val = [make_example(64, rng) for _ in range(600)]
val_x = torch.stack([to_tensor(im) for im, _ in val])
val_lb = [[float(v) for v in lb.split()] for _, lb in val]


# ---------- 00: el problema completo ----------
logits = model(img[None])[0]
box, score, pcls, pprob = predict(logits)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw=dict(width_ratios=[1, .8, 1, 1]))
ax[0].imshow(img_np); ax[0].set_title("1. Entra una imagen\n64 × 64 píxeles en color")
ax[1].axis("off")
ax[1].add_patch(FancyBboxPatch((.05, .2), .9, .6, boxstyle="round,pad=.03", fc="#dbe9f6", ec="#1f77b4", lw=2))
ax[1].text(.5, .5, "RED NEURONAL\n35 736 pesos\naprendidos", ha="center", va="center", fontsize=12, weight="bold")
ax[1].annotate("", (1.15, .5), (.95, .5), arrowprops=dict(arrowstyle="->", lw=2), xycoords="axes fraction")
ax[1].annotate("", (.05, .5), (-.15, .5), arrowprops=dict(arrowstyle="->", lw=2), xycoords="axes fraction")
obj = torch.sigmoid(logits[0]).numpy()
tabla(ax[2], obj, fmt="{:.2f}", cmap="viridis", vmin=0, vmax=1, fs=7)
ax[2].set_title("2. Sale una respuesta por casilla\n(aquí: «¿hay objeto?», 8 × 8)")
ax[3].imshow(img_np)
caja(ax[3], *box.tolist(), "lime", lw=3)
ax[3].set_title(f"3. Nos quedamos con la mejor:\n{NOMBRES[pcls]} ({pprob:.0%}), caja verde")
for a in (ax[0], ax[3]):
    a.axis("off")
save(fig, "00_problema.png")


# ---------- 01: píxeles ----------
r0, c0, n = 9, 14, 6
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw=dict(width_ratios=[1, 1.4]))
ax[0].imshow(img_np)
caja(ax[0], c0, r0, c0 + n, r0 + n, "black", lw=2)
ax[0].set_title("La imagen completa (64 × 64)\nel recuadro negro es la zona ampliada")
ax[1].imshow(img_np[r0:r0 + n, c0:c0 + n])
for (r, c), _ in np.ndenumerate(img_np[r0:r0 + n, c0:c0 + n, 0]):
    R, G, B = (int(v) for v in img_np[r0 + r, c0 + c])
    ax[1].text(c, r, f"R {R}\nG {G}\nB {B}", ha="center", va="center", fontsize=8,
               color="white" if R + G + B < 400 else "black")
ax[1].set_xticks(np.arange(-.5, n), []); ax[1].set_yticks(np.arange(-.5, n), []); ax[1].grid(color="k", lw=1)
ax[1].set_title(f"Ampliación de {n} × {n} píxeles: cada píxel = 3 números (0-255)")
save(fig, "01_pixeles.png")


# ---------- 02: los 3 canales ----------
fig, ax = plt.subplots(1, 4, figsize=(15, 4))
ax[0].imshow(img_np); ax[0].set_title("imagen en color\ntensor (3, 64, 64)")
for k, (nombre, cmap) in enumerate((("R (rojo)", "Reds_r"), ("G (verde)", "Greens_r"), ("B (azul)", "Blues_r"))):
    im = ax[k + 1].imshow(img[k].numpy(), cmap="gray", vmin=0, vmax=1)
    ax[k + 1].set_title(f"canal {k} = {nombre}\nimg[{k}] → matriz (64, 64)")
for a in ax:
    a.axis("off")
fig.colorbar(im, ax=ax, fraction=.015, label="valor (0 = negro, 1 = blanco)")
save(fig, "02_canales.png")


# ---------- 03: la neurona ----------
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
xs, ws, nom = (.9, .2, .1), (1.0, -.5, -.5), ("R", "G", "B")
for i, (x, wgt, nm) in enumerate(zip(xs, ws, nom)):
    y = 5 - i * 2
    ax.add_patch(plt.Circle((1, y), .5, fc=("#f4a6a0", "#a8dca8", "#a6c4f0")[i], ec="k"))
    ax.text(1, y, f"{nm}\n{x}", ha="center", va="center", fontsize=11)
    ax.annotate("", (5.4, 3), (1.5, y), arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
    ax.text(3.2, (y + 3) / 2 + .25, f"× {wgt:+}", fontsize=12, color="firebrick", weight="bold")
    ax.text(3.2, (y + 3) / 2 - .2, f"= {x * wgt:+.2f}", fontsize=10)
ax.add_patch(plt.Circle((6, 3), .7, fc="#fff3c4", ec="k"))
ax.text(6, 3, "Σ\n+ b", ha="center", va="center", fontsize=13)
ax.text(6, 1.8, "sesgo b = −0.3", ha="center", fontsize=10, color="firebrick")
ax.text(6, 4.2, "0.9 − 0.1 − 0.05 − 0.3\n= 0.45", ha="center", fontsize=10)
ax.annotate("", (8, 3), (6.7, 3), arrowprops=dict(arrowstyle="->", lw=2))
ax.add_patch(FancyBboxPatch((8, 2.4), 1.4, 1.2, boxstyle="round,pad=.05", fc="#e6dcf5", ec="k"))
ax.text(8.7, 3, "ReLU\nmax(0, ·)", ha="center", va="center", fontsize=11)
ax.annotate("", (11, 3), (9.5, 3), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(11.2, 3, "0.45\n¡se activa!", va="center", fontsize=12, weight="bold", color="green")
for xx, t in ((1, "Entradas (x)"), (3.5, "Pesos (w):\nlo que la red APRENDE"), (6, "Suma"), (8.7, "Activación"), (11.5, "Salida")):
    ax.text(xx, -.1, t, ha="center", fontsize=10, color="dimgray")
ax.set_title("Una neurona «detectora de rojo» mirando un píxel rojo [0.9, 0.2, 0.1]", fontsize=13)
save(fig, "03_neurona.png")


# ---------- 04: activaciones ----------
x = np.linspace(-6, 6, 300)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(x, np.maximum(0, x), lw=3); ax[0].set_title("ReLU(x) = max(0, x)\nnegativos → 0, positivos pasan igual")
ax[1].plot(x, 1 / (1 + np.exp(-x)), lw=3, color="tab:purple")
ax[1].set_title("sigmoide σ(x)\ncualquier número → entre 0 y 1")
for xv in (-2, 0, 2):
    yv = 1 / (1 + np.exp(-xv)); ax[1].plot(xv, yv, "ko"); ax[1].annotate(f"σ({xv}) = {yv:.2f}", (xv, yv), (xv + .3, yv - .12))
for a in ax[:2]:
    a.axhline(0, color="k", lw=.5); a.axvline(0, color="k", lw=.5); a.grid(alpha=.3); a.set_xlabel("x (lo que entra)")
lg = np.array([2.0, .1, -1.0]); pr = np.exp(lg) / np.exp(lg).sum()
pos = np.arange(3)
ax[2].bar(pos - .2, lg, .4, label="logits (salida en bruto)", color="lightgray", ec="k")
ax[2].bar(pos + .2, pr, .4, label="probabilidad (softmax)", color=COLOR_CLASE, ec="k")
for p, v in zip(pos, pr):
    ax[2].text(p + .2, v + .05, f"{v:.0%}", ha="center", weight="bold")
for p, v in zip(pos, lg):
    ax[2].text(p - .2, v + .05 if v > 0 else v / 2, f"{v}", ha="center")
ax[2].set_xticks(pos, NOMBRES); ax[2].axhline(0, color="k", lw=.5); ax[2].legend(fontsize=8)
ax[2].set_title("softmax: 3 números → 3 probabilidades\nque suman 1")
save(fig, "04_activaciones.png")


# ---------- 05: convolución paso a paso ----------
X = np.array([[1, 1, 1, .2, .2, .2]] * 5)
K = np.array([[1, 0, -1]] * 3)
Y = np.array([[(X[r:r + 3, c:c + 3] * K).sum() for c in range(4)] for r in range(3)])
fig, ax = plt.subplots(2, 3, figsize=(14, 8), gridspec_kw=dict(width_ratios=[6, 3, 4]))
for fila, c in enumerate((0, 1)):
    tabla(ax[fila, 0], X, cmap="Greys_r", vmin=0, vmax=1.3, resaltar=(0, c, 3, 3))
    ax[fila, 0].set_title(f"imagen (5 × 6): ventana 3 × 3 en columnas {c}-{c + 2}")
    tabla(ax[fila, 1], K, cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax[fila, 1].set_title("filtro (9 pesos)")
    parcial = np.full_like(Y, np.nan); parcial[0, :c + 1] = Y[0, :c + 1]
    tabla(ax[fila, 2], parcial if fila == 0 else Y, fmt="{:.1f}", cmap="viridis", vmin=0, vmax=2.4,
          resaltar=(0, c, 1, 1))
    cuenta = " + ".join(f"{v:g}×{k:g}" for v, k in zip(X[0, c:c + 3], K[0])) + f" = {(X[0, c:c + 3] * K[0]).sum():g}"
    ax[fila, 2].set_title(f"cada fila: {cuenta}\n3 filas → {Y[0, c]:.1f}", fontsize=10)
ax[1, 2].set_xlabel("…y así en todas las posiciones → FEATURE MAP\nvale mucho donde hay borde, 0 donde no", fontsize=11)
fig.suptitle("Convolución: el filtro se desliza, multiplica casilla a casilla y suma", fontsize=14)
save(fig, "05_convolucion.png")


# ---------- 05b: filtros sobre una imagen real ----------
gris = torch.nn.functional.pad(img.mean(0, keepdim=True)[None], (1, 1, 1, 1), mode="replicate")  # sin falsos bordes
filtros = {"borde vertical": [[1, 0, -1]] * 3, "borde horizontal": [[1] * 3, [0] * 3, [-1] * 3],
           "rojo − (verde+azul)/2": None}
fig, ax = plt.subplots(1, 4, figsize=(15, 4))
ax[0].imshow(img_np); ax[0].set_title("imagen")
for a, (nombre, k) in zip(ax[1:], filtros.items()):
    if k is None:
        m = img[0] - (img[1] + img[2]) / 2
    else:
        m = torch.nn.functional.conv2d(gris, torch.tensor(k, dtype=torch.float32)[None, None])[0, 0]
    a.imshow(m.abs().numpy(), cmap="magma"); a.set_title(f"filtro «{nombre}»\n(brillante = se activa)")
for a in ax:
    a.axis("off")
save(fig, "05b_filtros_a_mano.png")


# ---------- 05c: lo que la red aprendió de verdad (capa 1) ----------
W = model.backbone[0].weight.numpy()                # (16, 3, 3, 3)
mapas = model.backbone[:3](img[None])[0].numpy()    # conv + BN + ReLU → (16, 32, 32)
fig, ax = plt.subplots(2, 16, figsize=(18, 3.2))
for k in range(16):
    f = W[k].transpose(1, 2, 0); f = (f - f.min()) / (f.max() - f.min())
    ax[0, k].imshow(f); ax[1, k].imshow(mapas[k], cmap="magma")
    ax[0, k].set_title(f"filtro {k}", fontsize=8)
for a in ax.flat:
    a.axis("off")
fig.suptitle("Fila de arriba: los 16 filtros 3×3×3 APRENDIDOS por la 1.ª capa (como colores).  "
             "Fila de abajo: los 16 feature maps (32 × 32) que producen sobre nuestra imagen", fontsize=11)
save(fig, "05c_filtros_aprendidos.png")


# ---------- 05d: stride y padding ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
for a, s in zip(ax, (1, 2)):
    a.set_xlim(-.5, 7.5); a.set_ylim(7.5, -.5); a.set_aspect("equal")
    for r in range(8):
        for c in range(8):
            borde = r in (0, 7) or c in (0, 7)
            a.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc="#eeeeee" if borde else "#9ecae1", ec="gray"))
            if borde:
                a.text(c, r, "0", ha="center", va="center", color="gray", fontsize=9)
    centros = range(1, 7, s)
    for r in centros:
        for c in centros:
            a.plot(c, r, "o", color="crimson", ms=10)
    a.add_patch(Rectangle((-.5, -.5), 3, 3, fill=False, ec="crimson", lw=3))
    nsal = len(centros)
    a.set_title(f"stride = {s}: el filtro se centra en los puntos rojos\n"
                f"entrada 6 × 6 → salida {nsal} × {nsal}", fontsize=12)
    a.axis("off")
fig.suptitle("Azul = imagen 6 × 6,  gris = padding (relleno de ceros),  recuadro rojo = el filtro 3 × 3 en su primera posición",
             fontsize=11, y=1.06)
save(fig, "05d_stride_padding.png")


# ---------- 05e: campo receptivo ----------
x_rf = val_x[:64].clone().requires_grad_(True)
with torch.enable_grad():
    model.feature_maps(x_rf)[:, :, 4, 4].sum().backward()
g = x_rf.grad.abs().sum((0, 1)).numpy()
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
ax[0].imshow(np.full((64, 64), 1.0), cmap="gray", vmin=0, vmax=1)
rejilla(ax[0], color="lightgray")
for capa, (tam, col) in enumerate(zip((3, 7, 11, 19, 27, 43), plt.cm.plasma(np.linspace(0, .85, 6))), 1):
    caja(ax[0], 32 - tam // 2, 32 - tam // 2, 32 - tam // 2 + tam, 32 - tam // 2 + tam, col, lw=2.5,
         label=f"tras la capa {capa}: {tam} × {tam} px")
ax[0].plot(32, 32, "k*", ms=12)
ax[0].legend(loc="upper center", bbox_to_anchor=(.5, -.03), ncol=2, fontsize=9)
ax[0].set_title("Teoría: zona de la imagen que «ve» la casilla (4, 4)\ncrece capa tras capa")
ax[1].imshow(g, cmap="hot"); rejilla(ax[1], color="gray")
ax[1].set_title("Realidad (red entrenada): cuánto influye cada píxel\nen la casilla (4, 4). Brillante = mucho")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
save(fig, "05e_campo_receptivo.png")


# ---------- 06: batchnorm ----------
bn = model.backbone[1]
k = 3
z = model.backbone[0](val_x[:64])[:, k].flatten()          # salida de la conv, canal k, 64 imágenes
zn = (z - z.mean()) / z.std()                               # normalizado a mano
zbn = zn * bn.weight[k] + bn.bias[k]                        # reescalado con los 2 números aprendidos
zr = zbn.clamp(min=0)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
for a, v, t in zip(ax, (z, zn, zbn, zr), ("1. salida de la conv", "2. − media, ÷ desviación",
                                           f"3. × γ ({bn.weight[k]:.2f}) + β ({bn.bias[k]:.2f})", "4. ReLU")):
    a.hist(v.numpy(), bins=60, color="tab:blue")
    a.set_title(f"{t}\nmedia {v.mean():.2f}, desviación {v.std():.2f}", fontsize=10)
    a.axvline(0, color="k", lw=.8)
ax[0].set_ylabel("cuántos valores")
fig.suptitle(f"BatchNorm sobre el canal {k} de la 1.ª capa (64 imágenes × 32 × 32 valores)", fontsize=12, y=1.07)
save(fig, "06_batchnorm.png")


# ---------- 07a: arquitectura ----------
etapas = [("imagen", 3, 64), ("conv s2", 16, 32), ("conv s1", 16, 32), ("conv s2", 32, 16), ("conv s1", 32, 16),
          ("conv s2", 32, 8), ("conv s1", 32, 8), ("cabeza 1×1", 8, 8)]
fig, ax = plt.subplots(figsize=(16, 5))
ax.axis("off"); ax.set_xlim(-1, 41); ax.set_ylim(-10, 9)
x0 = 0
for i, (nombre, ch, sz) in enumerate(etapas):
    hgt, wid = sz / 64 * 14, ch / 32 * 2.5 + .4
    color = "#c7e9c0" if i == 0 else "#fdd0a2" if i == len(etapas) - 1 else "#c6dbef"
    for d in range(3):  # efecto "pila"
        ax.add_patch(Rectangle((x0 + d * .25, -hgt / 2 + d * .25), wid, hgt, fc=color, ec="k", lw=.8))
    ax.text(x0 + wid / 2, hgt / 2 + 1.2, nombre, ha="center", fontsize=10, weight="bold")
    ax.text(x0 + wid / 2, -hgt / 2 - 1.3, f"({ch}, {sz}, {sz})", ha="center", fontsize=10)
    if 0 < i < len(etapas) - 1:
        ax.text(x0 + wid / 2, -hgt / 2 - 2.4, "+BN+ReLU", ha="center", fontsize=8, color="dimgray")
    x0 += wid + 2.6
    if i < len(etapas) - 1:
        ax.annotate("", (x0 - .3, 0), (x0 - 2.1, 0), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(14, -9, "BACKBONE: extrae características  (el alto = tamaño espacial,  el ancho = número de canales)",
        ha="center", fontsize=11, color="#1f77b4")
ax.text(37, -9, "CABEZA: responde", ha="center", fontsize=11, color="#e6550d")
save(fig, "07a_arquitectura.png")


# ---------- 07b: el viaje de una imagen por la red ----------
acts, x_ = [], img[None]
for i, capa in enumerate(model.backbone):
    x_ = capa(x_)
    if isinstance(capa, torch.nn.ReLU):
        acts.append(x_[0].numpy())
out = model.head(x_)[0]
fig, ax = plt.subplots(8, 8, figsize=(14, 15))
ax[0, 0].imshow(img_np); ax[0, 0].set_title("imagen", fontsize=9)
for kk, nm in enumerate("RGB"):
    ax[0, kk + 1].imshow(img[kk], cmap="gray"); ax[0, kk + 1].set_title(f"canal {nm}", fontsize=9)
for f, a_ in enumerate(acts, 1):
    canales = np.argsort(-a_.reshape(len(a_), -1).max(1))[:8]   # los 8 canales que más se activan
    for j, c in enumerate(canales):
        ax[f, j].imshow(a_[c], cmap="magma"); ax[f, j].set_title(f"canal {c}", fontsize=8)
    ax[f, 0].set_ylabel(f"capa {f}\n{tuple(a_.shape)}", fontsize=9, rotation=0, ha="right", va="center")
p5 = torch.sigmoid(out[:5]); pc = torch.softmax(out[5:], 0)
titulos = ["¿objeto?", "dx", "dy", "w", "h"] + [f"P({n})" for n in NOMBRES]
for j, (m, t) in enumerate(zip(list(p5) + list(pc), titulos)):
    ax[7, j].imshow(m.numpy(), cmap="viridis", vmin=0, vmax=1); ax[7, j].set_title(t, fontsize=10, weight="bold")
ax[7, 0].set_ylabel("cabeza\n(8, 8, 8)", fontsize=9, rotation=0, ha="right", va="center")
for a in ax.flat:
    a.set_xticks([]); a.set_yticks([])
    if not a.images:
        a.axis("off")
fig.suptitle("El viaje de la imagen por la red: cada fila es la salida de una capa (8 de sus canales)\n"
             "la imagen se hace más pequeña y más «abstracta»; la última fila es la respuesta", fontsize=13)
save(fig, "07b_viaje_imagen.png")


# ---------- 08a: la cuadrícula y la casilla responsable ----------
tgt = code(cx, cy, w, h, int(cls))
fig, ax = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw=dict(width_ratios=[1.2, 1]))
ax[0].imshow(img_np); rejilla(ax[0], color="dimgray")
col, row = int(cx // 8), int(cy // 8)
ax[0].add_patch(Rectangle((col * 8 - .5, row * 8 - .5), 8, 8, fc="yellow", alpha=.35, ec="red", lw=2.5))
caja(ax[0], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, "blue", lw=2, ls="--")
ax[0].plot(cx - .5, cy - .5, "r*", ms=16)
ax[0].text(38, 4, f"casilla responsable (amarilla):\nfila {row}, columna {col} = píxeles {col * 8}-{col * 8 + 8} × {row * 8}-{row * 8 + 8}\n"
           f"el ★ está a {cx - col * 8:g} px de su borde izquierdo\n→ dx = {cx - col * 8:g}/8 = {tgt[1, row, col]:.3f}\n"
           f"y a {cy - row * 8:g} px de su borde superior\n→ dy = {cy - row * 8:g}/8 = {tgt[2, row, col]:.2f}",
           fontsize=9, va="top", bbox=dict(fc="white", ec="red"))
ax[0].annotate("", (cx - .5, cy - .5), (38, 10), arrowprops=dict(arrowstyle="->", color="red"))
ax[0].set_xticks(np.arange(4, 64, 8), range(8)); ax[0].set_yticks(np.arange(4, 64, 8), range(8))
ax[0].set_xlabel("columna de la cuadrícula"); ax[0].set_ylabel("fila de la cuadrícula")
ax[0].set_title(f"Etiqueta «{label}» → centro (★) en la casilla fila {row}, columna {col}\n"
                f"caja azul: {w:g} × {h:g} px  →  w = h = {w:g}/64 = {w / 64:g}", fontsize=11)
ax[1].axis("off")
sub = ax[1].inset_axes([0, 0, 1, 1]); sub.axis("off")
nombres_t = ["0: ¿objeto?", "1: dx", "2: dy", "3: w", "4: h", "5: clase"]
for k_ in range(6):
    a = ax[1].inset_axes([(k_ % 3) * .34, .52 - (k_ // 3) * .52, .31, .44])
    tabla(a, tgt[k_].numpy(), fmt="{:.2g}", cmap="Blues", vmin=0, vmax=1.2, fs=6, resaltar=(row, col, 1, 1))
    a.set_title(nombres_t[k_], fontsize=10)
ax[1].set_title("El objetivo que construye code(): tensor (6, 8, 8)\ntodo 0 salvo la casilla responsable", fontsize=11)
save(fig, "08a_cuadricula.png")


# ---------- 08b: lo que responde la red y cómo se elige la caja ----------
boxes, scores, probs = decode(logits)
best = int(scores.argmax())
fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
ax[0].imshow(img_np)
for b, s in sorted(zip(boxes, scores), key=lambda t: float(t[1])):
    caja(ax[0], *b.tolist(), "lime" if s > .5 else "gray", lw=3 if s > .5 else .7, alpha=1 if s > .5 else .5)
ax[0].set_title("decode(): las 64 casillas proponen una caja cada una\n(gris: casillas que dicen «aquí no hay nada»; verde: la segura)")
tabla(ax[1], scores.reshape(8, 8).numpy(), fmt="{:.2f}", cmap="viridis", vmin=0, vmax=1, fs=8,
      resaltar=(best // 8, best % 8, 1, 1))
ax[1].set_title("probabilidad de objeto por casilla (sigmoide del canal 0)\nnos quedamos con la mayor")
ax[2].bar(range(3), probs[best].numpy(), color=COLOR_CLASE, ec="k")
for i_, v in enumerate(probs[best]):
    ax[2].text(i_, float(v) + .02, f"{float(v):.4f}", ha="center")
ax[2].set_xticks(range(3), NOMBRES); ax[2].set_ylim(0, 1.1)
ax[2].set_title(f"forma en ESA casilla (softmax de los canales 5-7)\n→ {NOMBRES[pcls]}")
ax[0].set_xticks([]); ax[0].set_yticks([])
save(fig, "08b_decodificar.png")


# ---------- 09: las pérdidas ----------
p = np.linspace(.005, .995, 300)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(p, -np.log(p), lw=3, label="había objeto: −log(p)")
ax[0].plot(p, -np.log(1 - p), lw=3, label="no había: −log(1 − p)")
for pv in (.1, .9):
    ax[0].plot(pv, -np.log(pv), "ko"); ax[0].annotate(f"{-np.log(pv):.2f}", (pv, -np.log(pv)), (pv + .03, -np.log(pv) + .2))
ax[0].set_xlabel("p = probabilidad de objeto que da la red"); ax[0].set_ylabel("error"); ax[0].legend()
ax[0].set_title("BCE: «¿hay objeto?»")
e = np.linspace(-.3, .3, 200)
ax[1].plot(e, e ** 2, lw=3, color="tab:purple"); ax[1].plot(.05, .0025, "ko")
ax[1].annotate("w predicho 0.30, real 0.25\nerror 0.05² = 0.0025", (.05, .0025), (-.05, .05))
ax[1].set_xlabel("predicho − real"); ax[1].set_title("MSE: «¿dónde y qué tamaño?»")
ax[2].plot(p, -np.log(p), lw=3, color="tab:green")
for pv, (ox, oy) in zip((.99, .83, 1 / 3, .05), ((-.25, .9), (-.25, .6), (.03, .15), (.03, .15))):
    ax[2].plot(pv, -np.log(pv), "ko")
    ax[2].annotate(f"p={pv:.2f} → {-np.log(pv):.2f}", (pv, -np.log(pv)), (pv + ox, -np.log(pv) + oy),
                   arrowprops=dict(arrowstyle="-", color="gray"))
ax[2].set_xlabel("probabilidad que la red da a la forma CORRECTA"); ax[2].set_title("CE: «¿qué forma?»")
for a in ax:
    a.grid(alpha=.3)
save(fig, "09_perdidas.png")


# ---------- 11a: descenso de gradiente ----------
def pasos(lr, n=6, w0=1.0):
    ws_ = [w0]
    for _ in range(n):
        ws_.append(ws_[-1] - lr * 2 * (2 * ws_[-1] - 6) * 2)
    return np.array(ws_)

wr = np.linspace(-1.5, 7.5, 300)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for a, lr, t in zip(ax, (.1, .01, .26), ("lr = 0.1: bien", "lr = 0.01: demasiado pequeño (lento)",
                                          "lr = 0.26: demasiado grande (rebota y se aleja)")):
    a.plot(wr, (2 * wr - 6) ** 2, color="lightgray", lw=3)
    ws_ = pasos(lr, n={.1: 2, .01: 6}.get(lr, 4))
    ls_ = (2 * ws_ - 6) ** 2
    a.plot(ws_, ls_, "o-", color="crimson")
    for i_, (wv, lv) in enumerate(zip(ws_, ls_)):
        a.annotate(str(i_), (wv, lv), (wv + .15, lv + 3), fontsize=9)
    a.set_title(t); a.set_xlabel("valor del peso w"); a.set_ylim(-3, 90); a.axvline(3, ls=":", color="green")
ax[0].set_ylabel("pérdida = (2w − 6)²")
ax[0].text(3.1, 70, "mínimo\nw = 3", color="green")
fig.suptitle("Descenso de gradiente con UN peso: cada punto rojo es un paso (empezando en w = 1)", fontsize=12, y=1.03)
save(fig, "11a_descenso.png")


# ---------- 11b: el bucle de entrenamiento ----------
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 7)
cajas = [(1.5, 5.5, "1. Lote de 64 imágenes\n+ respuestas correctas", "#c7e9c0"),
         (6, 5.5, "2. model(images)\nla red predice\n(64, 8, 8, 8)", "#c6dbef"),
         (10.5, 5.5, "3. compute_loss(...)\nun número: cuánto\nse equivoca", "#fdd0a2"),
         (10.5, 1.5, "4. loss.backward()\ngradiente de cada\nuno de los 35 736 pesos", "#fcbba1"),
         (6, 1.5, "5. optimizer.step()\ncada peso se mueve\nun poquito «cuesta abajo»", "#dadaeb"),
         (1.5, 1.5, "6. optimizer.zero_grad()\nborrar gradientes\ny siguiente lote", "#eeeeee")]
for xx, yy, t, c in cajas:
    ax.add_patch(FancyBboxPatch((xx - 1.4, yy - .9), 2.8, 1.8, boxstyle="round,pad=.1", fc=c, ec="k"))
    ax.text(xx, yy, t, ha="center", va="center", fontsize=10)
for (x1, y1, *_), (x2, y2, *_) in zip(cajas, cajas[1:] + cajas[:1]):
    dx_, dy_ = np.sign(x2 - x1) * 1.55, np.sign(y2 - y1) * 1.05
    ax.annotate("", (x2 - dx_, y2 - dy_), (x1 + dx_, y1 + dy_), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(6, 3.5, "47 vueltas = 1 época\n40 épocas ≈ 1 900 vueltas", ha="center", va="center", fontsize=12,
        weight="bold", color="firebrick")
ax.set_title("El bucle de entrenamiento (función train())", fontsize=13)
save(fig, "11b_bucle.png")


# ---------- 11c: antes y después de entrenar ----------
torch.manual_seed(0)
crudo = MultiClassDetector(deep=True).eval()
idx = [0, 1, 2, 3, 4, 5]
fig, ax = plt.subplots(2, 6, figsize=(16, 6))
for fila, (m, nombre) in enumerate(((crudo, "SIN entrenar (pesos al azar)"), (model, "ENTRENADA"))):
    for j, i_ in enumerate(idx):
        b, s, c_, pc_ = predict(m(val_x[i_:i_ + 1])[0])
        lx, ly, lw_, lh_, lc = val_lb[i_]
        ax[fila, j].imshow(val[i_][0])
        caja(ax[fila, j], lx - lw_ / 2, ly - lh_ / 2, lx + lw_ / 2, ly + lh_ / 2, "blue", ls="--")
        caja(ax[fila, j], *b.tolist(), "lime" if c_ == lc else "red", lw=2.5)
        ax[fila, j].set_title(f"{NOMBRES[c_]} ({pc_:.0%})\nIoU {float(iou(b, torch.tensor([[lx - lw_ / 2, ly - lh_ / 2, lx + lw_ / 2, ly + lh_ / 2]]))):.2f}",
                              fontsize=9)
        ax[fila, j].axis("off")
    ax[fila, 0].text(-8, 32, nombre, rotation=90, va="center", ha="right", fontsize=11, weight="bold")
fig.suptitle("Misma red, mismas imágenes de validación. Azul discontinuo = caja real; verde/rojo = caja predicha "
             "(rojo si la forma es incorrecta)", fontsize=11)
save(fig, "11c_antes_despues.png")


# ---------- 12: IoU ----------
casos = [((10, 10, 30, 30), (20, 10, 40, 30)), ((10, 10, 30, 30), (14, 13, 34, 33)), ((10, 10, 30, 30), (11, 10, 31, 30))]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
for a, (A, B) in zip(ax, casos):
    a.set_xlim(5, 45); a.set_ylim(38, 5); a.set_aspect("equal")
    a.add_patch(Rectangle(A[:2], A[2] - A[0], A[3] - A[1], fc="tab:blue", alpha=.3, ec="tab:blue", lw=2))
    a.add_patch(Rectangle(B[:2], B[2] - B[0], B[3] - B[1], fc="tab:orange", alpha=.3, ec="tab:orange", lw=2))
    ix = max(0, min(A[2], B[2]) - max(A[0], B[0])); iy = max(0, min(A[3], B[3]) - max(A[1], B[1]))
    a.add_patch(Rectangle((max(A[0], B[0]), max(A[1], B[1])), ix, iy, fill=False, hatch="///", ec="k"))
    inter = ix * iy; union = 400 + 400 - inter
    a.set_title(f"intersección {inter}   unión 400 + 400 − {inter} = {union}\nIoU = {inter}/{union} = {inter / union:.2f}"
                + ("  ✔ > 0.5" if inter / union > .5 else "  ✘ < 0.5"), fontsize=10)
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("Azul = caja real (20 × 20), naranja = caja predicha, rayado = intersección", fontsize=12)
save(fig, "12_iou.png")


# ---------- 13: projector casero (PCA de las 32 características) ----------
feats = model.feature_maps(val_x)                                       # (600, 32, 8, 8)
cols_ = [min(int(l[0] // 8), 7) for l in val_lb]; rows_ = [min(int(l[1] // 8), 7) for l in val_lb]
F = torch.stack([feats[i, :, r, c] for i, (r, c) in enumerate(zip(rows_, cols_))])  # (600, 32)
F = F - F.mean(0)
P = (F @ torch.linalg.svd(F, full_matrices=False).Vh[:2].T).numpy()   # proyección en 2 dimensiones (PCA)
clase = np.array([int(l[4]) for l in val_lb])
pal = np.array(COLORS) / 255
color_id = np.array([np.argmin(((np.array(COLORS) - np.asarray(im)[int(l[1]), int(l[0])]) ** 2).sum(1))
                     for (im, _), l in zip(val, val_lb)])
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for k_ in range(3):
    ax[0].scatter(*P[clase == k_].T, s=12, color=COLOR_CLASE[k_], label=NOMBRES[k_])
ax[0].legend(); ax[0].set_title("Coloreado por FORMA → 3 grupos separados")
for k_ in range(5):
    ax[1].scatter(*P[color_id == k_].T, s=12, color=pal[k_])
ax[1].set_title("Mismos puntos, coloreados por el COLOR de la figura → mezclados")
for a in ax:
    a.set_xlabel("dirección principal 1"); a.set_ylabel("dirección principal 2"); a.grid(alpha=.3)
fig.suptitle("Cada punto = una imagen de validación: sus 32 características (casilla responsable) aplastadas a 2D",
             fontsize=12)
save(fig, "13_projector.png")
