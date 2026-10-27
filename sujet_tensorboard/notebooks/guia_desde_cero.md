# Guía desde cero: entender el notebook de la opción B

> **Cómo leer esta guía.** Empieza por el principio y no te saltes capítulos: cada uno usa lo del anterior.
> Todos los números de los ejemplos están calculados de verdad (y muchos salen tal cual en el notebook).
> En VS Code, abre la vista previa con **Ctrl + Shift + V** para ver las tablas y fórmulas bien.

---

## 0. El problema en una frase

Tenemos imágenes pequeñas (64 × 64 píxeles) con **una figura** dentro: un círculo, un cuadrado o un triángulo,
de un color cualquiera, en un sitio cualquiera. Queremos un programa que, al ver la imagen, diga:

1. **dónde** está la figura (dibujando un rectángulo a su alrededor, la "caja");
2. **qué** figura es (círculo, cuadrado o triángulo).

No le vamos a escribir las reglas a mano ("si hay una curva, es un círculo…"). Vamos a darle **miles de ejemplos
con la respuesta correcta** y dejar que **aprenda solo** las reglas. Eso es el *aprendizaje automático*,
y la herramienta que aprende es una **red neuronal**.

---

## 1. ¿Qué es una imagen para un ordenador?

Una imagen es una cuadrícula de puntitos llamados **píxeles**. Nuestras imágenes tienen 64 columnas y 64 filas.

Cada píxel tiene un color, y un color se describe con **3 números**: cuánto **rojo**, cuánto **verde**
y cuánto **azul** (R, G, B), cada uno entre 0 (nada) y 255 (máximo).

| color | R | G | B |
|---|---|---|---|
| negro | 0 | 0 | 0 |
| blanco | 255 | 255 | 255 |
| rojo de nuestras figuras | 230 | 75 | 60 |
| gris claro del fondo | 235 | 235 | 235 |

Así que una imagen de 64 × 64 son en realidad **64 × 64 × 3 = 12 288 números**. Para el ordenador no hay
"círculos" ni "colores": solo hay esos 12 288 números. La red tendrá que descubrir, a partir de ellos, dónde
está la figura y qué forma tiene.

En el notebook, la función `to_tensor` divide todo por 255 para que los números queden **entre 0 y 1**
(0,9 en lugar de 230). Las redes aprenden mejor con números pequeños.

---

## 2. ¿Qué es un tensor?

Un **tensor** es simplemente **una caja de números ordenados en varias dimensiones**. Es la palabra que usa
PyTorch para todo dato.

| nombre | dimensiones | ejemplo | `shape` (forma) |
|---|---|---|---|
| escalar | 0 | `7` | `()` |
| vector | 1 | `[0.9, 0.3, 0.2]` (un píxel RGB) | `(3,)` |
| matriz | 2 | una imagen en blanco y negro 64×64 | `(64, 64)` |
| tensor 3D | 3 | una imagen en color | `(3, 64, 64)` |
| tensor 4D | 4 | un **lote** de 64 imágenes en color | `(64, 3, 64, 64)` |

La **forma** (`shape`) dice cuántos elementos hay en cada dimensión. **Leer formas es la habilidad n.º 1**
para entender el notebook. Las que verás:

| tensor | forma | qué significa |
|---|---|---|
| una imagen | `(3, 64, 64)` | 3 colores × 64 filas × 64 columnas |
| un lote de imágenes | `(64, 3, 64, 64)` | 64 imágenes |
| las *feature maps* | `(32, 8, 8)` | 32 "mapas" de 8 × 8 (lo veremos) |
| la salida de la red | `(8, 8, 8)` | 8 números por cada una de las 8 × 8 casillas |
| el objetivo (respuesta correcta) | `(6, 8, 8)` | 6 números por casilla |

**Cómo se lee un elemento:** `target[:, 1, 3]` significa "todos los canales (`:`), fila 1, columna 3" → los 6 números
de la casilla (1, 3). `target[0]` es "el canal 0 entero" → un mapa de 8 × 8.

> ¿Por qué `(3, 64, 64)` y no `(64, 64, 3)`? Las imágenes suelen guardarse como (alto, ancho, color), pero PyTorch
> quiere el color primero. Por eso `to_tensor` hace `.permute(2, 0, 1)`: reordena las dimensiones.

---

## 3. La neurona: el ladrillo básico

Una **neurona artificial** es una cuenta muy simple:

1. recibe varios números de entrada;
2. multiplica cada uno por un **peso**;
3. lo suma todo y añade un número extra, el **sesgo** (*bias*);
4. pasa el resultado por una **función de activación** (por ejemplo ReLU, ver el capítulo 4).

$$\text{salida} = \text{activación}(w_1 x_1 + w_2 x_2 + w_3 x_3 + b)$$

### Ejemplo: una neurona "detectora de rojo"

Entradas: los valores R, G, B de un píxel. Pesos: `w = [1.0, −0.5, −0.5]`, sesgo `b = −0.3`.

* **Píxel rojo** `[0.9, 0.2, 0.1]`: `0.9×1 + 0.2×(−0.5) + 0.1×(−0.5) − 0.3 = 0.9 − 0.1 − 0.05 − 0.3 = 0.45` → **se activa**.
* **Píxel gris** `[0.9, 0.9, 0.9]`: `0.9 − 0.45 − 0.45 − 0.3 = −0.3` → con ReLU queda en **0**: no se activa.

Con esos pesos, la neurona "se enciende" con el rojo y se queda apagada con el gris.

### ¿Qué es un peso? (la idea más importante de todas)

Los **pesos** (y sesgos) son los números que deciden lo que hace la neurona. Con otros pesos, la misma neurona
detectaría el azul o los píxeles oscuros. **Al principio los pesos son aleatorios** (la red no sabe nada) y
**aprender = ir ajustando los pesos** hasta que la red dé buenas respuestas.

Nuestra red tiene **35 736 pesos**. Nadie los elige a mano: los encuentra el entrenamiento (capítulo 11).
En el código se llaman "parámetros" (`model.parameters()`).

### ¿Qué es una "red"?

Muchas neuronas organizadas en **capas**: las salidas de una capa son las entradas de la siguiente.
Las primeras capas detectan cosas simples (bordes, colores), y las siguientes combinan esas detecciones para
reconocer cosas más complejas (esquinas → formas).

---

## 4. Funciones de activación: ReLU, sigmoide y softmax

### ReLU

$$\text{ReLU}(x) = \max(0, x)$$

Deja pasar los positivos y pone los negativos a 0.

| x | −3 | −0.3 | 0 | 0.45 | 2 |
|---|---|---|---|---|---|
| ReLU(x) | 0 | 0 | 0 | 0.45 | 2 |

**¿Para qué sirve algo tan simple?** Sin ella, una capa es "multiplicar y sumar" y otra capa encima también es
"multiplicar y sumar"… y dos "multiplicar y sumar" seguidos equivalen a **uno solo**. Apilar 10 capas no serviría
de nada. La ReLU "dobla" el cálculo y permite que cada capa añada capacidad de verdad. Por eso se dice que es una
**no linealidad**. En el notebook va detrás de cada convolución (`conv_block`).

### Sigmoide

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Transforma **cualquier número en un número entre 0 y 1**.

| x | −5 | −2 | 0 | 2 | 5 |
|---|---|---|---|---|---|
| σ(x) | 0.007 | 0.119 | 0.5 | 0.881 | 0.993 |

La usamos cuando queremos una respuesta entre 0 y 1: "¿hay un objeto?" (una probabilidad), y la posición/tamaño
de la caja (que guardamos como fracciones entre 0 y 1).

### Softmax

Transforma **una lista de números en probabilidades que suman 1**. Se usa para elegir entre varias opciones.

La red da 3 números para las 3 formas, por ejemplo `[2.0, 0.1, −1.0]` (círculo, cuadrado, triángulo).
El softmax calcula $e^x$ de cada uno y divide por la suma:

| | círculo | cuadrado | triángulo | suma |
|---|---|---|---|---|
| número de la red ("logit") | 2.0 | 0.1 | −1.0 | |
| $e^x$ | 7.389 | 1.105 | 0.368 | 8.862 |
| probabilidad | **0.834** | 0.125 | 0.042 | 1 |

→ "Es un círculo, con un 83 % de confianza".

> **Logit** = el número "en bruto" que sale de la red, *antes* de la sigmoide o el softmax. En el código, la salida
> de `model(...)` son logits; `decode()` les aplica sigmoide/softmax.

---

## 5. La convolución (el corazón de la red)

### El problema

Podríamos conectar cada uno de los 12 288 números de la imagen a cada neurona, pero serían muchísimos pesos,
y la red tendría que aprender por separado "un borde arriba a la izquierda" y "un borde abajo a la derecha".
Un borde es un borde, esté donde esté.

### La idea

Una **convolución** usa un **filtro** pequeño (aquí de **3 × 3**) que se **desliza** por toda la imagen.
En cada posición, multiplica los 9 píxeles que tiene debajo por sus 9 pesos y suma: es una neurona que mira
solo un trocito de 3 × 3, y **la misma neurona (los mismos pesos) se reutiliza en todas las posiciones**.

### Ejemplo numérico: detectar un borde vertical

Un trozo de imagen (en gris, 5 filas × 6 columnas): a la izquierda fondo claro (1), a la derecha figura oscura (0.2):

```
1   1   1   0.2 0.2 0.2
1   1   1   0.2 0.2 0.2
1   1   1   0.2 0.2 0.2
1   1   1   0.2 0.2 0.2
1   1   1   0.2 0.2 0.2
```

Filtro (pesos):

```
1  0  -1
1  0  -1
1  0  -1
```

Este filtro calcula "(lo de la izquierda) − (lo de la derecha)". Lo colocamos sobre las columnas 0–2:
cada fila da `1×1 + 1×0 + 1×(−1) = 0`, total **0** (zona uniforme, no hay borde).
Lo movemos una columna (columnas 1–3): cada fila da `1×1 + 1×0 + 0.2×(−1) = 0.8`, total **2.4** (¡borde!).
Haciendo esto en todas las posiciones obtenemos:

```
0   2.4  2.4  0
0   2.4  2.4  0
0   2.4  2.4  0
```

El resultado es un **mapa** que vale mucho justo donde está el borde y 0 en el resto.
Eso es una **feature map** (mapa de características): "dónde aparece la característica que busca este filtro".

> En nuestra red **nadie escribe** estos filtros: los 9 números de cada filtro son **pesos** que se aprenden.
> La red descubre sola qué filtros le son útiles (bordes, esquinas, colores…).

### Varios filtros = varios canales

Una capa no tiene un filtro, sino muchos. `Conv2d(3, 16, ...)` tiene **16 filtros** → produce **16 mapas**
(16 "canales"). Y como la imagen de entrada tiene 3 canales (R, G, B), cada filtro es en realidad de
**3 × 3 × 3** (mira los 3 colores a la vez).

Cuenta de pesos de esa capa: 16 filtros × (3 × 3 × 3) = **432 pesos** + 16 sesgos = 448. (Es exactamente lo que da
PyTorch para la primera capa de nuestro modelo.)

### Stride (paso) y padding (relleno)

* **Stride = 2**: el filtro avanza de 2 en 2 píxeles en lugar de 1 → el mapa de salida mide **la mitad**
  (64 → 32). Es la forma de "resumir" la imagen poco a poco.
* **Padding = 1**: se añade un borde de ceros alrededor de la imagen para que el filtro pueda centrarse también
  en los píxeles del borde.

### Campo receptivo

Tras una convolución 3 × 3, cada valor "ve" 3 × 3 píxeles. Si encima pones otra, cada valor ve 3 × 3 valores
de la capa anterior, que a su vez veían 3 × 3 píxeles → ve una zona más grande. Cuantas más capas, **más grande
es la zona de la imagen que influye en cada valor** (el *campo receptivo*). Al final, cada casilla de 8 × 8
"ve" un buen trozo de la imagen, suficiente para abarcar una figura entera.

---

## 6. BatchNorm (normalización por lotes)

Después de cada convolución, los números pueden salir muy grandes o muy pequeños, y cambian mucho de una
capa a otra mientras la red aprende. Eso hace el aprendizaje inestable.

La **BatchNorm** coge, para cada canal, los valores de todas las imágenes del lote, les **resta la media y divide
por la desviación**, de modo que queden "centrados en 0 y de tamaño 1". Luego los reescala con 2 números
aprendidos por canal (por eso aparecen pesos `backbone.1.weight` y `backbone.1.bias` de 16 elementos).

Resultado: un entrenamiento más estable y rápido. El TP exigía "normalización por lotes + no linealidad" en
cada capa: por eso cada bloque es **Conv → BatchNorm → ReLU**.

> **`model.train()` y `model.eval()`.** Al entrenar, la BatchNorm usa la media del lote actual. Al evaluar
> usa una media "aprendida" durante el entrenamiento (así el resultado de una imagen no depende de las otras
> del lote). Por eso el código cambia de modo antes de entrenar y antes de evaluar.

---

## 7. Nuestra red completa

```
imagen 64×64 en color                          (3, 64, 64)
│
├─ BACKBONE (extrae características)
│   Conv 3×3 stride 2 + BN + ReLU        →   (16, 32, 32)   la imagen se reduce a la mitad
│   Conv 3×3 stride 1 + BN + ReLU        →   (16, 32, 32)   misma talla, una capa más de "reflexión"
│   Conv 3×3 stride 2 + BN + ReLU        →   (32, 16, 16)
│   Conv 3×3 stride 1 + BN + ReLU        →   (32, 16, 16)
│   Conv 3×3 stride 2 + BN + ReLU        →   (32,  8,  8)
│   Conv 3×3 stride 1 + BN + ReLU        →   (32,  8,  8)   ← "feature maps": 32 números por casilla
│
└─ CABEZA (da la respuesta)
    Conv 1×1                             →   ( 8,  8,  8)   ← 8 números por casilla
```

La imagen se ha reducido **8 veces** (64 → 8): es el **stride total** `STRIDE = 8`. Cada casilla de la
cuadrícula 8 × 8 corresponde a un cuadrado de 8 × 8 píxeles de la imagen original.

**La cabeza (convolución 1 × 1)**: un filtro de 1 × 1 no mira a los vecinos; en cada casilla, toma sus 32 números
y calcula 8 combinaciones (8 neuronas). Es decir: **en cada casilla, 8 neuronas leen las 32 características y
dan la respuesta**. Los 8 números son:

| canal | significado | después de… |
|---|---|---|
| 0 | ¿hay un objeto cuyo centro está en esta casilla? | sigmoide → probabilidad |
| 1, 2 | `dx`, `dy`: dónde está el centro **dentro** de la casilla | sigmoide → entre 0 y 1 |
| 3, 4 | `w`, `h`: ancho y alto de la caja, como fracción de la imagen | sigmoide → entre 0 y 1 |
| 5, 6, 7 | puntuación de círculo, cuadrado, triángulo | softmax → probabilidades |

Los canales 0–4 son los del TP; los 5–7 son **lo nuevo de la opción B**.

### ¿De dónde salen los 35 736 pesos?

| capa | cálculo | pesos |
|---|---|---|
| Conv 3→16 | 16 × 3×3×3 + 16 | 448 |
| BN 16 | 2 × 16 | 32 |
| Conv 16→16 | 16 × 16×3×3 + 16 | 2 320 |
| BN 16 | | 32 |
| Conv 16→32 | 32 × 16×3×3 + 32 | 4 640 |
| BN 32 | | 64 |
| Conv 32→32 (×3) | 3 × (32 × 32×3×3 + 32) | 27 744 |
| BN 32 (×3) | | 192 |
| Cabeza 32→8 | 8 × 32 + 8 | 264 |
| **total** | | **35 736** |

El modelo del TP (3 convoluciones, 8→16→32 canales, cabeza de 5) tiene 6 309.

---

## 8. Cómo una red "encuentra" un objeto: la cuadrícula

La red no da una única caja: da una predicción **en cada una de las 64 casillas**. La regla del juego es:

> **La casilla que contiene el centro de la figura es la "responsable"**: ella debe decir "aquí hay un objeto"
> y dar la caja y la forma. Las otras 63 casillas deben decir "aquí no hay nada".

### Ejemplo real del notebook

La etiqueta del generador es `25.0 14.0 16 16 1`: un **cuadrado** (clase 1) de **16 × 16** px con centro en (25, 14).

1. Centro en unidades de casilla: `25 / 8 = 3.125` y `14 / 8 = 1.75`.
2. Casilla responsable = parte entera → **columna 3, fila 1**.
3. Posición dentro de la casilla = parte decimal → `dx = 0.125`, `dy = 0.75`.
4. Tamaño como fracción de la imagen: `16 / 64 = 0.25`.
5. El **objetivo** en la casilla (1, 3) es `[1, 0.125, 0.75, 0.25, 0.25, 1]`. En las demás, todo 0.

Eso hace la función `code()`, y es lo que imprime la sección 3 del notebook.

El camino inverso, `decode()`, va de la salida de la red a píxeles:
`cx = (3 + 0.125) × 8 = 25` ✔, `w = 0.25 × 64 = 16` ✔. Luego, de (centro, tamaño) a esquinas:
`x1 = 25 − 8 = 17`, `x2 = 25 + 8 = 33`, etc.

### Dar la respuesta final: `predict()`

1. `decode()` convierte las 64 casillas en 64 cajas + 64 probabilidades "¿objeto?" + 64 tríos de probabilidades de forma.
2. Nos quedamos con **la casilla con la mayor probabilidad de objeto** (la función `nms` ordena por puntuación;
   como hay una sola figura, basta con quedarse con la primera).
3. La forma predicha es la de mayor probabilidad **en esa casilla**.

> **NMS (Non-Maximum Suppression)**: cuando varias cajas se solapan mucho, quedarse solo con la de mayor
> puntuación. Aquí, con una sola figura, solo sirve para coger la mejor; sería importante con varias figuras.

---

## 9. ¿Cómo sabe la red que se equivoca? La función de pérdida

La **pérdida** (*loss*) es **un solo número que mide cuánto se equivoca la red**. 0 = perfecto; cuanto mayor,
peor. Todo el entrenamiento consiste en **hacer bajar este número**. Nuestra pérdida suma tres partes:

### Parte 1: "¿hay objeto?" → entropía cruzada binaria (BCE)

Para cada casilla, si la red da probabilidad `p` de objeto:
* si **sí** había objeto, el error es $-\log(p)$;
* si **no** había, el error es $-\log(1 - p)$.

| situación | p | error |
|---|---|---|
| había objeto, la red dice 0.9 | 0.9 | 0.105 (bien) |
| había objeto, la red dice 0.1 | 0.1 | 2.30 (muy mal) |
| no había, la red dice 0.1 | 0.1 | 0.105 (bien) |

Problema: 63 casillas vacías por 1 con objeto. Una red "perezosa" que siempre dijera "no hay nada" acertaría
63 de 64. Por eso el error de la casilla con objeto se **multiplica por 10** (`pos_weight=10`).

### Parte 2: "¿dónde y de qué tamaño?" → error cuadrático medio (MSE)

Solo en la casilla responsable: la media de los errores al cuadrado entre `(dx, dy, w, h)` predichos y reales.
Ej.: predice `w = 0.30` y era `0.25` → error `0.05² = 0.0025`. Como son errores pequeños, se multiplica por 10
para que pese lo suficiente.

### Parte 3 (nueva): "¿qué forma?" → entropía cruzada (CE)

Solo en la casilla responsable: $-\log(\text{probabilidad que la red dio a la forma correcta})$.

| la red da a la forma correcta… | error |
|---|---|
| 0.99 | 0.01 |
| 0.83 | 0.19 |
| 1/3 (al azar) | 1.10 |
| 0.01 | 4.6 |

Solo se calcula en la casilla con objeto: en las vacías no hay ninguna forma que adivinar.

**Total**: `pérdida = BCE + 10 × MSE + 1 × CE`.
La red sin entrenar tiene una pérdida de ~3.4; al final del entrenamiento, ~0.002.

---

## 10. Datos de entrenamiento y de validación

* **Entrenamiento**: 3 000 imágenes con las que la red aprende.
* **Validación**: 600 imágenes **distintas** que la red **nunca usa para aprender**. Sirven para medir si la red
  ha aprendido de verdad o solo ha memorizado. (Si funcionara muy bien en entrenamiento y mal en validación,
  estaría **memorizando**: eso se llama *sobreajuste*.)

Las imágenes se generan con una **semilla** (`seed`): un número que fija el azar, para que cada vez que ejecutes
el notebook salgan exactamente las mismas imágenes y los mismos resultados.

---

## 11. ¿Cómo aprende? El descenso de gradiente

### La intuición

Imagina que estás en una montaña con niebla y quieres bajar al valle (= pérdida mínima). No ves el valle, pero
sí notas **hacia dónde baja el suelo bajo tus pies**. Das un paso en esa dirección, vuelves a tantear, otro paso…
Cada peso de la red es una dirección en la que te puedes mover; el **gradiente** te dice, para cada peso,
"si lo aumentas un poco, ¿la pérdida sube o baja, y cuánto?".

### Ejemplo con un solo peso

Red mínima: `predicción = w × x`. Tenemos `x = 2` y la respuesta correcta es `6` (el peso ideal sería 3).
Pérdida = `(predicción − 6)²`. Empezamos con `w = 1`, y el paso (*learning rate*) es `0.1`.

| paso | w | predicción | pérdida | gradiente `2·(pred − 6)·x` | nuevo w = w − 0.1 × gradiente |
|---|---|---|---|---|---|
| 1 | 1.00 | 2.00 | 16.00 | −16.0 | 2.60 |
| 2 | 2.60 | 5.20 | 0.64 | −3.2 | 2.92 |
| 3 | 2.92 | 5.84 | 0.026 | −0.64 | 2.98 |

En 3 pasos, `w` va de 1 a casi 3 y la pérdida de 16 a casi 0. **Nuestra red hace exactamente esto, pero con
35 736 pesos a la vez.**

### Las 4 líneas que lo hacen (función `train()` de `src/train_multiclass.py`)

```python
optimizer.zero_grad()                         # 1. borra los gradientes del paso anterior (si no, se sumarían)
loss = compute_loss(model(images), targets)   # 2. la red predice y medimos la pérdida
loss.backward()                               # 3. PyTorch calcula el gradiente de los 35 736 pesos (¡automáticamente!)
optimizer.step()                              # 4. mueve cada peso un poquito en la dirección que baja la pérdida
```

* `backward()` es la magia de PyTorch: como ha ido "apuntando" todas las cuentas hechas, puede calcular
  solo todas las derivadas (esto se llama *retropropagación*).
* **Adam** (el `optimizer`) es una versión mejorada del descenso de gradiente: ajusta el tamaño del paso para cada
  peso y tiene "inercia" (si va varios pasos en la misma dirección, acelera).
* El **learning rate** (`lr = 0.003`) es el tamaño del paso. Muy grande → saltas por encima del valle;
  muy pequeño → tardas una eternidad. El `StepLR` lo divide por 10 en la época 30 para afinar al final.

### Lote y época

* Un **lote** (*batch*) = 64 imágenes que se procesan juntas; los pesos se corrigen **una vez por lote**
  (con el error medio de las 64). 3 000 / 64 ≈ **47 correcciones**.
* Una **época** = pasar una vez por las 3 000 imágenes. Hacemos **40 épocas** → unas 1 900 correcciones en total.
* `shuffle=True`: las imágenes se mezclan en cada época para que la red no aprenda nada del orden.

### `torch.no_grad()`

Cuando solo queremos ver qué responde la red (evaluar, dibujar), no hace falta apuntar las cuentas para el
gradiente: `with torch.no_grad():` lo desactiva, y así va más rápido y gasta menos memoria.

---

## 12. ¿Cómo medimos si funciona? Las métricas

### IoU (Intersection over Union)

Compara la caja predicha con la real:

$$\text{IoU} = \frac{\text{área de la intersección}}{\text{área de la unión}}$$

Ejemplo: dos cajas de 10 × 10 desplazadas 5 px en horizontal.
Intersección = 5 × 10 = 50. Unión = 100 + 100 − 50 = 150 (se resta la intersección para no contarla dos veces).
IoU = 50 / 150 = **0.33**.

* IoU = 1 → cajas idénticas; IoU = 0 → no se tocan.
* Se considera "detección buena" si **IoU > 0.5**.

### Precisión de la forma (*accuracy*)

% de imágenes en que la forma predicha es la correcta.

### Matriz de confusión

Tabla que cruza **forma real** (filas) con **forma predicha** (columnas). Ejemplo inventado:

| real ↓ / predicha → | círculo | cuadrado | triángulo |
|---|---|---|---|
| círculo | **195** | 3 | 2 |
| cuadrado | 1 | **198** | 1 |
| triángulo | 0 | 4 | **196** |

La diagonal (en negrita) son los aciertos. Fuera de la diagonal, los errores: "3 círculos se tomaron por cuadrados".
* **Recall** de "cuadrado" (por fila): de los 200 cuadrados reales, ¿cuántos reconoció? 198 / 200 = 99 %.
* **Precisión** de "cuadrado" (por columna): de las 205 veces que dijo "cuadrado", ¿cuántas acertó? 198 / 205 = 96.6 %.

En nuestro notebook la matriz es perfecta (todo en la diagonal): 100 % de formas correctas.

---

## 13. TensorBoard: el panel de control

Mientras la red entrena, solo vemos unas líneas de texto. **TensorBoard** es una página web que muestra
**gráficas e imágenes del entrenamiento en directo**.

### Cómo funciona

1. En el código, un `SummaryWriter` (el `writer`) escribe en unos archivos de "logs" todo lo que le pedimos.
2. El programa `tensorboard` lee esos archivos y los dibuja en http://localhost:6006.

Para abrirlo, copia en una terminal PowerShell el comando que imprime la celda 3 del notebook, por ejemplo:

```
& "C:\Users\lucas\Downloads\ISIS\NNR\rnn\Scripts\python.exe" -m tensorboard.main --logdir "C:\Users\lucas\AppData\Local\Temp\tb_runs\option_B" --port 6006
```

y abre http://localhost:6006 en el navegador. Cada subcarpeta (modelo) aparece como un **run** a la izquierda.

### Qué escribe el notebook y dónde verlo

| función en el código | pestaña | qué ves | qué te dice |
|---|---|---|---|
| `add_scalar` | **SCALARS** | curvas: pérdida, IoU, % de formas, `lr` por época | si la red aprende (pérdida ↓, IoU ↑) y cuándo deja de mejorar |
| `add_figure` | **IMAGES** | ejemplos, matriz de confusión, predicciones, peores casos | resultados visuales |
| `add_graph` | **GRAPHS** | el esquema de la red (doble clic para abrir los bloques) | cómo está construida |
| `add_histogram` | **HISTOGRAMS / DISTRIBUTIONS** | cómo se reparten los pesos de cada capa, época a época | si los pesos se mueven (= la red aprende) |
| `add_embedding` | **PROJECTOR** | 600 puntos en 3D, uno por figura, coloreables por forma | si la red separa bien las formas |
| `add_hparams` | **HPARAMS** | tabla comparativa de los 3 modelos | qué modelo es mejor |

### El PROJECTOR, explicado

Para cada imagen, la cabeza decide la forma a partir de **32 números** (las características de la casilla
responsable). Son como el "carné de identidad" de la figura, pero en 32 dimensiones, que no podemos dibujar.
TensorBoard las **reduce a 3 dimensiones** (con métodos llamados PCA o t-SNE) para poder dibujarlas como puntos.
Si coloreas por `label` y ves **tres nubes separadas**, significa que el backbone ya ha "ordenado" las figuras por
forma, y que a la cabeza solo le queda decir "esta nube es la de los círculos". Por eso la clasificación es tan fácil aquí.

---

## 14. El notebook, sección por sección

| sección | qué hace | conceptos de esta guía |
|---|---|---|
| 1. Mise en place | importa todo, fija constantes y la carpeta de TensorBoard | cap. 1, 2, 13 |
| 2. Nouvelles images | genera y muestra 10 ejemplos | cap. 0, 1 |
| 3. Codage | convierte una etiqueta en objetivo `(6, 8, 8)` | cap. 8 |
| 4. Le réseau | crea la red y comprueba las formas y el número de pesos | cap. 3, 5, 6, 7 |
| 5. La perte | calcula la pérdida de la red sin entrenar | cap. 9 |
| 6. Entraînement | genera 3 000 + 600 imágenes y entrena 40 épocas | cap. 10, 11, 13 |
| 7. Évaluation | IoU, formas, matriz de confusión, por tamaño, ejemplos, feature maps, projector | cap. 5, 12, 13 |
| 8. Comparaison | entrena el detector del TP y una versión ligera, y compara | cap. 7, 12 |
| 9. Sauvegarde | guarda los pesos en `detector_multiclass.pt` | cap. 3 |

### Los resultados y qué significan

| modelo | IoU | IoU > 0.5 | forma correcta | pesos |
|---|---|---|---|---|
| Detector del TP (solo caja) | 0.780 | 96.2 % | – | 6 309 |
| Nuestro modelo, backbone pequeño | 0.782 | 96.2 % | 100 % | 6 408 |
| Nuestro modelo, backbone profundo | **0.948** | **100 %** | **100 %** | 35 736 |

1. **Añadir la forma no estropea la caja**: con el mismo backbone, 0.780 frente a 0.782. Y solo cuesta 99 pesos
   más (las 3 neuronas nuevas de la cabeza: 3 × 32 + 3).
2. **Reconocer la forma es fácil aquí**: incluso el modelo pequeño acierta el 100 %. Las figuras están sobre
   un fondo liso, con bordes nítidos.
3. **Lo que mejora la caja es el backbone más profundo** (más capas → más campo receptivo y más pesos): IoU de 0.78 a 0.95.
4. **Límites**: imágenes muy simples (una figura, sin rotación, sin ruido, siempre tan ancha como alta).
   En fotos reales no tendríamos estos números.

---

## 15. Explicarlo en 2 minutos (guion)

> "Queremos que una red encuentre una figura en una imagen de 64×64 y diga si es un círculo, un cuadrado o un
> triángulo, con un color al azar para que no pueda fijarse solo en el color.
>
> La red tiene dos partes. El **backbone** es una pila de convoluciones: filtros de 3×3 que se deslizan por la
> imagen y detectan bordes, esquinas, colores; cada bloque reduce la imagen a la mitad, hasta una cuadrícula de
> 8×8 con 32 características por casilla. La **cabeza** es una convolución 1×1 que, en cada casilla, da 8 números:
> ¿hay un objeto aquí?, dónde está su centro, su tamaño, y una puntuación para cada forma.
>
> Para entrenar, convertimos cada etiqueta en lo que la casilla responsable —la que contiene el centro— debería
> responder. La pérdida suma tres errores: ¿objeto? (entropía cruzada binaria), la caja (error cuadrático) y,
> lo nuevo, la forma (entropía cruzada). El descenso de gradiente ajusta los 35 000 pesos para que esa pérdida baje.
>
> Respecto al TP, solo hemos cambiado tres cosas: 3 salidas más en la cabeza, un valor más en el objetivo y un
> término más en la pérdida. Resultado: 100 % de formas correctas y un IoU medio de 0,95. La comparación muestra
> que añadir la forma no estropea la caja y que lo que mejora la caja es el backbone más profundo.
> Lo hemos seguido todo con TensorBoard: curvas, imágenes y el projector, donde se ve que las tres formas
> quedan bien separadas."

---

## Glosario rápido

| término | en pocas palabras |
|---|---|
| píxel | un punto de la imagen, con 3 números (R, G, B) |
| tensor | caja de números en varias dimensiones; `shape` = su forma |
| neurona | suma ponderada + sesgo + activación |
| peso / parámetro | número que la red ajusta al aprender |
| sesgo (*bias*) | número que se suma en la neurona |
| capa | grupo de neuronas que trabajan a la vez |
| ReLU | `max(0, x)`: la no linealidad |
| sigmoide | lleva cualquier número a [0, 1] |
| softmax | convierte varios números en probabilidades que suman 1 |
| logit | salida en bruto de la red, antes de sigmoide/softmax |
| convolución | filtro pequeño que se desliza por la imagen |
| filtro / kernel | los pesos de la convolución (3×3) |
| canal | uno de los mapas que salen de una convolución (uno por filtro) |
| feature map | mapa que indica dónde aparece una característica |
| stride | cuánto avanza el filtro; con 2, la salida mide la mitad |
| padding | borde de ceros añadido alrededor |
| campo receptivo | zona de la imagen que influye en un valor |
| BatchNorm | recentra y reescala los valores para estabilizar el aprendizaje |
| backbone | parte de la red que extrae características |
| cabeza (*head*) | última capa, que da la respuesta |
| casilla responsable | la que contiene el centro de la figura |
| codificar / decodificar | etiqueta → objetivo de la red / salida de la red → caja en píxeles |
| pérdida (*loss*) | número que mide el error; se quiere bajar |
| BCE / CE / MSE | errores para sí/no, para elegir entre clases, y para valores numéricos |
| gradiente | para cada peso, hacia dónde mover para bajar la pérdida |
| `backward()` | calcula todos los gradientes automáticamente |
| optimizador (Adam) | aplica las correcciones a los pesos |
| learning rate (`lr`) | tamaño del paso de corrección |
| lote (*batch*) | grupo de imágenes procesadas a la vez (64) |
| época | una pasada por todas las imágenes de entrenamiento |
| validación | imágenes nunca usadas para aprender, para medir de verdad |
| sobreajuste | memorizar en vez de aprender |
| IoU | intersección / unión de dos cajas |
| NMS | quedarse con la mejor caja entre las que se solapan |
| matriz de confusión | tabla real × predicha |
| precisión / recall | aciertos sobre lo predicho / sobre lo real |
| semilla (*seed*) | número que fija el azar para repetir resultados |
| TensorBoard | página web que muestra las gráficas del entrenamiento |
