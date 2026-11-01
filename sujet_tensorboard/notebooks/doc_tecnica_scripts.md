# Documentación técnica: los scripts de la opción B

> Qué es: la referencia técnica de **qué archivo hace qué**, con qué funciones, qué entra y qué sale de cada una,
> y cómo se encadenan para que la opción B funcione. Para entender los conceptos (convolución, pérdida, IoU…)
> ver [guia_detallada.md](guia_detallada.md).

---

## 1. Mapa de archivos

```
sujet_tensorboard/
├── src/
│   ├── dataset_generator.py     (TP, dado)      generador: siempre la misma forma, siempre rojo
│   ├── dataset_generator_2.py   (dado, usado)   generador opción B: forma y color aleatorios  ◄── datos
│   ├── dataset_generator_3.py   (dado, no usado) varias figuras por imagen (opción C)
│   ├── detector.py              (TP)            Detector, code/decode, iou, nms, pérdida del TP
│   ├── train.py                 (TP)            train/evaluate del TP
│   ├── visualize.py             (TP)            show_images, show_feature_maps, show_detections
│   ├── detector_multiclass.py   (NUEVO)         todo el modelo de la opción B                  ◄── modelo
│   └── train_multiclass.py      (NUEVO)         entrenamiento, evaluación, curvas, script CLI  ◄── entrenamiento
├── notebooks/
│   ├── demo.ipynb               (TP)            el detector original
│   ├── mini_projet_option_B.ipynb (NUEVO)       demostración completa de la opción B (soutenance)
│   ├── guia_desde_cero.md / guia_detallada.md (NUEVO)   guías de conceptos
│   ├── guia_img/hacer_figuras.py (NUEVO)        genera las ilustraciones de la guía
│   └── doc_tecnica_scripts.md   (NUEVO)         este documento
├── figures/*.png                (NUEVO)         figuras del informe, las escribe el notebook
├── rapport/rapport_option_B.{md,pdf} (NUEVO)    informe
└── detector_multiclass.pt       (NUEVO)         pesos entrenados (los escribe el notebook)
```

**Qué se creó y qué se reutilizó**

* **Dado por el sujet y reutilizado sin cambios de lógica** (solo se añadieron comentarios en español):
  `dataset_generator_2.py`, `detector.py` (`STRIDE`, `iou`, `nms`, `Detector`), `visualize.py` (`show_feature_maps`).
* **Creado para la opción B:** `detector_multiclass.py`, `train_multiclass.py`, el notebook, el informe y las figuras.

### Dependencias entre módulos

```
dataset_generator_2.make_example ─────────────┐
                                              ▼
detector.STRIDE, iou, nms ──► detector_multiclass ──► train_multiclass ──► mini_projet_option_B.ipynb
detector.Detector (comparación) ──────────────────────────────────────────►        ▲
visualize.show_feature_maps ───────────────────────────────────────────────────────┘
```

Solo se usan las bibliotecas del TP: `torch`, `numpy`, `Pillow`, `matplotlib`, `tensorboard`.

---

## 2. Del TP a la opción B: qué cambia exactamente

| pieza | TP (`detector.py`) | opción B (`detector_multiclass.py`) |
|---|---|---|
| etiqueta | `cx cy w h` | `cx cy w h class_id` |
| `code()` | objetivo `(5, 8, 8)` | objetivo `(6, 8, 8)`: + canal 5 = número de clase |
| red, cabeza | `Conv2d(32, 5, 1)` | `Conv2d(32, 5 + 3, 1)`: + 3 logits de clase |
| red, backbone | 3 convs stride 2 (3→8→16→32) | `deep=True`: 6 convs (3→16→16→32→32→32→32); `deep=False` = el del TP |
| `decode()` | cajas + puntuaciones | cajas + puntuaciones + **probabilidades de clase** (softmax) |
| `compute_loss()` | BCE + 10·MSE | BCE + 10·MSE + **1·CE** (entropía cruzada de clase) |
| `predict()` | caja + puntuación | caja + puntuación + **clase + probabilidad** |
| evaluación | IoU | IoU + **precisión de forma + matriz de confusión + IoU por clase** |

Lo que **no** cambia: la cuadrícula 8×8 (`STRIDE = 8`), la forma de codificar la caja (dx, dy, w, h),
`iou()`, `nms()` y el `pos_weight=10` de la BCE. Se importan directamente de `detector.py`.

---

## 3. `src/dataset_generator_2.py` — los datos

| función | entrada | salida | qué hace |
|---|---|---|---|
| `draw_shape(draw, shape, box, color)` | dibujo PIL, `"circle"/"square"/"triangle"`, `(x1,y1,x2,y2)`, color RGB | — | dibuja la figura rellena con `color` y contorno casi negro |
| `make_example(size, rng)` | tamaño (64), `random.Random` | `(PIL.Image, "cx cy w h class_id")` | **una** imagen + su etiqueta |
| `generate(output, size, train, val, seed)` | carpeta, cantidades, semilla | archivos `.png` + `.txt` | guarda un dataset en disco (lo usa el script CLI) |

**La aleatorización que impide «hacer trampa» con el color** (dentro de `make_example`):

```python
class_id = rng.randrange(len(SHAPES))   # forma al azar: 0, 1 o 2
shape = SHAPES[class_id]
color = rng.choice(COLORS)              # color al azar entre 5, sorteado APARTE de la forma
```

Como la forma y el color se sortean por separado, cada color aparece con las tres formas en la misma proporción:
saber el color no da ninguna información sobre la clase. El resto: fondo gris aleatorio (225–250), lado de 10 a 21 px,
posición aleatoria dentro de la imagen, `w == h` siempre.

---

## 4. `src/detector_multiclass.py` — el modelo

Constantes: `CLASSES = ("circle", "square", "triangle")` (mismo orden que el generador), `NUM_CLASSES = 3`.
Importa del TP: `STRIDE`, `iou`, `nms`.

### `code(cx, cy, w, h, class_id, size=64) -> Tensor (6, G, G)`
Etiqueta en píxeles → objetivo de entrenamiento.
```python
gx, gy = cx / size * grid, cy / size * grid                     # centro en unidades de casilla
col, row = min(int(gx), grid - 1), min(int(gy), grid - 1)       # casilla responsable
target[:, row, col] = [1, gx - col, gy - row, w / size, h / size, class_id]
```
El `min(..., grid - 1)` evita salirse de la cuadrícula si el centro cae justo en el borde 64.

### `decode(logits, size=64) -> (boxes (G*G, 4), scores (G*G,), class_probs (G*G, C))`
Salida de la red de **una** imagen `(8, 8, 8)` → 64 cajas `[x1, y1, x2, y2]` en píxeles, 64 puntuaciones
(sigmoide del canal 0) y 64 tríos de probabilidades (softmax de los canales 5–7, `dim=0` = sobre las clases).

### `read_label(path)` y `MultiShapeDataset(folder)`
Leen el dataset **en disco** (`.png` + `.txt`). `__getitem__` devuelve `(imagen (3, 64, 64) en [0, 1], code(...))`.
Solo los usa el script CLI; el notebook genera los datos en memoria.

### `conv_block(c_in, c_out, stride) -> [Conv2d 3×3, BatchNorm2d, ReLU]`
El bloque básico (padding 1: con stride 1 conserva el tamaño, con stride 2 lo divide por 2).

### `class MultiClassDetector(nn.Module)`
```python
MultiClassDetector(num_classes=3, deep=True)
  .backbone      # nn.Sequential de conv_blocks
  .head          # nn.Conv2d(32, 5 + num_classes, kernel_size=1)
  .feature_maps(x)   # (B, 3, 64, 64) -> (B, 32, 8, 8)
  .forward(x)        # (B, 3, 64, 64) -> (B, 8, 8, 8)   logits sin activar
```

| `deep` | backbone | parámetros |
|---|---|---|
| `True` | (3→16 s2, 16→16 s1), (16→32 s2, 32→32 s1), (32→32 s2, 32→32 s1) | 35 736 |
| `False` | 3→8 s2, 8→16 s2, 16→32 s2 (idéntico al TP) | 6 408 |

### `compute_loss(logits, target, box_weight=10, class_weight=1) -> Tensor escalar`
```python
obj_loss   = BCE_with_logits(logits[:, 0], target[:, 0], pos_weight=10)   # 64 casillas
positive   = target[:, 0] > 0.5                                           # máscara (B, G, G), una casilla por imagen
box_loss   = MSE(sigmoid(logits[:, 1:5])[positive], target[:, 1:5][positive])
class_loss = cross_entropy(logits[:, 5:][positive], target[:, 5][positive].long())
return obj_loss + 10 * box_loss + 1 * class_loss
```
Detalles técnicos:
* `.permute(0, 2, 3, 1)` pasa de `(B, C, G, G)` a `(B, G, G, C)` para poder indexar con la máscara `(B, G, G)`
  y obtener `(B, C)`: una fila por imagen.
* `cross_entropy` recibe **logits** (aplica el softmax internamente) y la clase como entero (`.long()`).
* La CE solo se calcula en la casilla positiva: en las demás no hay clase que aprender.

### `predict(logits, size=64) -> (box (4,), score float, class_id int, class_prob float)`
`decode` → `nms(boxes, scores, 0.5)[0]` (índice de la casilla de mayor puntuación) → clase = `argmax` de las
probabilidades **en esa casilla**.

---

## 5. `src/train_multiclass.py` — entrenamiento y evaluación

| función | entrada | salida | qué hace |
|---|---|---|---|
| `train(model, loader, optimizer)` | red, `DataLoader`, optimizador | pérdida media (float) | una época: `model.train()`, y por lote `zero_grad → loss → backward → step` |
| `target_to_label(target, size)` | objetivo `(6, G, G)` | `(caja real [x1,y1,x2,y2], clase real)` | inverso de `code()`, para comparar con la predicción |
| `evaluate(model, loader, size)` | red, `DataLoader`, 64 | `dict` (abajo) | `model.eval()`, `predict()` en cada imagen y métricas |
| `plot_history(history, path)` | dict de listas por época | figura (y PNG) | pérdida + métricas de validación |
| `main()` | argumentos CLI | `detector_multiclass.pt`, `courbes_multiclass.png`, logs TB | script completo desde archivos |

Claves que devuelve `evaluate`:

| clave | tipo | significado |
|---|---|---|
| `mean_iou` | float | IoU medio |
| `iou_gt_05` | float | proporción de IoU > 0,5 |
| `accuracy` | float | proporción de formas correctas |
| `detection_ok` | float | IoU > 0,5 **y** forma correcta |
| `confusion` | `LongTensor (3, 3)` | `confusion[real, predicha]` |
| `iou_per_class` | lista de 3 floats | IoU medio por forma real |
| `ious` | `Tensor (N,)` | IoU de cada imagen, **en el orden del loader** (sin `shuffle`) |

---

## 6. Cómo los usa el notebook `mini_projet_option_B.ipynb`

| sección | funciones llamadas | produce |
|---|---|---|
| 1. Mise en place | imports, `to_tensor` (definida en el notebook) | carpeta de logs `RUN_DIR`, comando TensorBoard |
| 2. Images | `make_example` ×10 (semilla 123) | `figures/exemples_dataset.png`, TB IMAGES |
| 3. Codage | `code` | imprime la casilla y los 6 valores |
| 4. Réseau | `MultiClassDetector`, `Detector` | formas de salida, nº de parámetros |
| 5. Perte | `code`, `compute_loss` | pérdida de la red sin entrenar (≈ 3,4) |
| 6. Entraînement | `make_dataset` (notebook) → `make_example` + `code`; `fit` (notebook) → `train`, `evaluate` | 3 000 + 600 imágenes en memoria (semillas 7 y 8), 40 épocas, TB SCALARS/HISTOGRAMS/GRAPHS, `figures/courbes_apprentissage.png` |
| 7. Évaluation | `evaluate`, `predict`, `iou`, `show_feature_maps` | métricas, `confusion_iou.png`, `predictions.png`, `erreurs.png`, TB PROJECTOR |
| 8. Comparaison | `fit` sobre `TPAsMultiClass(Detector())` y `MultiClassDetector(deep=False)` | tabla de 3 modelos, `comparaison.png`, TB HPARAMS |
| 9. Sauvegarde | `torch.save(model.state_dict())` | `detector_multiclass.pt` |

Dos piezas definidas en el notebook:

* **`fit(model, epochs, writer)`**: Adam (lr 0,003) + `StepLR` (÷10 a los 3/4 de las épocas); en cada época llama a
  `train` y `evaluate` y escribe en TensorBoard (`add_scalar`, `add_histogram`; `add_graph` una vez).
* **`TPAsMultiClass`**: envuelve el `Detector` del TP y le añade 3 canales de ceros para que tenga 8 salidas.
  Así se reutilizan `fit`/`evaluate` sin tocarlas; los ceros no tienen pesos, por lo que solo se entrena la caja.

---

## 7. Cómo ejecutarlo

Desde la carpeta `sujet_tensorboard`, con el entorno `rnn`:

```powershell
# A. El notebook (lo que se presenta): Run All en VS Code, o en línea de comandos
..\rnn\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\mini_projet_option_B.ipynb

# B. El script, sin notebook: genera data_multiclass/ en disco la primera vez y entrena
..\rnn\Scripts\python.exe -m src.train_multiclass --epochs 40

# C. TensorBoard (el notebook imprime este comando con la ruta exacta)
..\rnn\Scripts\python.exe -m tensorboard.main --logdir "$env:TEMP\tb_runs\option_B" --port 6006
```

* Los logs de TensorBoard van a la carpeta temporal del sistema (`%TEMP%\tb_runs\option_B\<fecha>`), no al proyecto,
  porque TensorBoard funciona mal en carpetas sincronizadas.
* **Reproducibilidad:** semillas fijas (`random.Random(7/8/123)` para los datos, `torch.manual_seed(0)` para los
  pesos). En CPU, dos ejecuciones dan los mismos números.
* El notebook **sobrescribe** `figures/*.png` y `detector_multiclass.pt`.

Para reutilizar el modelo entrenado:

```python
from src.detector_multiclass import MultiClassDetector, predict
model = MultiClassDetector()
model.load_state_dict(torch.load("detector_multiclass.pt"))
model.eval()
box, score, class_id, prob = predict(model(image[None])[0])   # image: (3, 64, 64) en [0, 1]
```

---

## 8. Límites conocidos del código

* `predict` se queda con **una** caja por imagen: no sirve para varias figuras (opción C).
* Una sola caja por casilla y sin ancres: dos centros en la misma casilla no se podrían separar (opción A).
* `w == h` siempre en los datos: la red nunca ha visto rectángulos alargados.
* `MultiShapeDataset` y `main()` solo se usan en el modo script; el notebook no los necesita.
