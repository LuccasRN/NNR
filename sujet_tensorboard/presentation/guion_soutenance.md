# Guion de la soutenance: Option B (≈ 9 min 30)

Para cada diapositiva tienes tres partes:
- **Qué muestra**: en español, explica cada elemento de la diapositiva para que sepas qué estás señalando.
- **Qué decir**: el texto en francés que puedes decir casi tal cual. Es el mismo que está en las notas del orador del `.pptx`.
- **Transición**: la frase que enlaza con la diapositiva siguiente. Es lo que evita que la exposición suene a temas sueltos.

## El hilo conductor (memorízalo)

Toda la presentación cuenta **una sola historia** con 4 pasos. Si te pierdes, vuelve a esto:

1. **El problema** (diapos 1–4). En el TP el réseau solo decía *dónde* está un círculo rojo. Yo quiero que diga *dónde* y *qué forma*, sin hacer trampa con el color.
2. **La herramienta** (diapos 5–6). Un réseau de convoluciones que convierte la imagen en una cuadrícula de 8×8 casillas.
3. **Cómo aprende** (diapos 7–8):
   - le digo qué debería responder (la *cible*);
   - mido cuánto se equivoca (la *perte*).
4. **¿Funciona y por qué?** (diapos 9–10):
   - resultados;
   - qué cambio es el responsable de la mejora.

Cada transición responde a la pregunta que deja abierta la diapositiva anterior.

---

## 1. Titre (0:00 – 0:20)

**Qué muestra:** el título, el subtítulo «Option B», tu nombre y las 3 formas.

**Qué decir:**
> Bonjour, je m'appelle Lucas Rodríguez. Dans le TP, le réseau trouvait où était un cercle rouge. Pour l'option B, je lui ai appris à trouver la figure **et** à dire si c'est un cercle, un carré ou un triangle, quelle que soit sa couleur.

**Transición:** « Pour y arriver, je suis parti du TP. »

---

## 2. Du TP à l'option B (0:20 – 1:05)

**Qué muestra:**
- arriba, ejemplos de imágenes generadas: cualquier forma en cualquier color;
- abajo, 3 tarjetas con los 3 únicos cambios respecto al TP.

**Qué decir:**
> Je suis parti du détecteur du TP et je n'ai changé que trois choses. Un : les données. La forme et la couleur sont tirées au hasard et l'étiquette contient maintenant le numéro de la forme. Deux : le réseau. La tête donne 3 valeurs de plus par case, une par forme. Trois : la perte, avec un terme de plus pour la forme. Tout le reste vient du TP : la grille 8×8, le codage de la boîte, l'IoU et la NMS.

**Transición:** « Voici où se trouvent ces changements dans le code. »

---

## 3. Les fichiers du projet (1:05 – 2:00)

**Qué muestra:**
- 3 tarjetas, una por archivo `.py`, con sus funciones;
- abajo, el notebook que lo une todo.

**Qué decir:**
> Le générateur `dataset_generator_2` était fourni avec le sujet, je l'ai utilisé tel quel. J'ai écrit deux scripts. `detector_multiclass.py` contient le modèle : `code()` fabrique la réponse attendue, `decode()` la relit, `MultiClassDetector` est le réseau, `compute_loss()` la perte, et `predict()` donne la boîte et la forme. `train_multiclass.py` contient l'entraînement et l'évaluation. Le notebook enchaîne ces fonctions.

⚠️ No digas que escribiste el generador: venía con el sujet.

**Transición:** « Avant de parler du réseau, un point important sur les données : la couleur. »

---

## 4. La couleur ne doit rien dire sur la forme (2:00 – 3:00)

**Qué muestra:**
- a la izquierda, las 2 líneas de código que sortean forma y color **por separado**;
- a la derecha, la proyección: cada punto es una imagen. Arriba están los puntos antes de entrenar y abajo después. La columna izquierda está coloreada por color y la derecha por forma.
- Después de entrenar, la columna «por forma» muestra 3 grupos limpios y la columna «por color» está mezclada.

**Qué decir:**
> Si les cercles étaient toujours rouges, le réseau pourrait tricher en regardant la couleur. Le générateur tire la forme et la couleur séparément, donc la couleur ne donne aucune information sur la forme. Les premières couches utilisent la couleur pour séparer la figure du fond, mais pour la forme elle ne sert à rien. À droite, chaque point est une image. Avant l'entraînement, rien n'est organisé. Après, il y a trois groupes, un par forme, et dans chaque groupe toutes les couleurs sont mélangées.

**Transición:** « Voyons maintenant comment le réseau transforme l'image. »

---

## 5. Le réseau (3:00 – 4:15)

**Qué muestra:**
- **El diagrama:** cada bloque es el resultado de una capa. La altura indica el tamaño (64, 32, 16, 8) y la anchura el número de canales. `s2` significa stride 2 y `s1` stride 1.
- **Tarjeta «Conv 3×3»:** un filtro de 3×3 píxeles se desliza por la imagen. Allí donde encuentra su motivo (un borde, una esquina, un color), da un valor alto.
- **Tarjeta «Stride 2 : réduire»:** stride 2 significa que el filtro avanza de 2 en 2 píxeles, así que la salida es la mitad de grande. Tres veces: 64 → 32 → 16 → 8. Al final queda una cuadrícula de 8×8 y cada casilla representa 8×8 píxeles de la imagen.
- **Tarjeta «Stride 1 : voir plus large»:** el *champ réceptif* es la zona de la imagen que una casilla final «ve».
  - Con el réseau del TP, cada casilla solo veía 15 píxeles, y las figuras grandes miden hasta 21: la casilla no veía la figura entera.
  - Una conv stride 1 no reduce el tamaño, pero cada capa más amplía la visión. Con las 3 que añadí, cada casilla ve 43 px y abarca la figura completa.

**Qué decir:**
> Le réseau prend l'image 64×64 et la transforme en une grille de 8×8 cases. Chaque convolution 3×3 fait glisser un petit filtre et détecte des motifs : bords, coins, couleurs. Les convolutions de stride 2 avancent de deux pixels en deux pixels : elles divisent la taille par deux, 64, 32, 16, 8. À la fin, chaque case représente un carré de 8 pixels de l'image.
> Le problème du TP : une case finale ne voyait que 15 pixels autour d'elle, moins que les plus grandes figures, qui font 21 pixels. J'ai donc ajouté à chaque étage une convolution de stride 1. Elle ne réduit pas l'image, mais chaque case voit maintenant 43 pixels, donc la figure entière.
> Enfin la tête, une convolution 1×1, transforme les 32 valeurs de chaque case en 8 réponses : 5 pour la boîte, 3 pour la forme.

**Transición:** « Concrètement, voici ce que voient ces couches. »

---

## 6. Ce que voient les convolutions (4:15 – 5:00)

**Qué muestra:** un cuadrado rojo que atraviesa el réseau entrenado.
- **Fila 1:** salidas de la capa 1, con bordes y relleno.
- **Fila 2:** capa 3, con esquinas y lados.
- **Fila 3:** capa 6, ya en 8×8. El cuadrado no se ve, pero se ilumina la zona del centro.
- **Última fila:** las respuestas de la cabeza. «objet ?» se enciende en **una sola** casilla, y en esa casilla P(carré) es alta.

**Qué decir:**
> Voici ce qui sort de chaque couche pour un carré rouge. Personne n'a écrit ces filtres : ils sont appris. La couche 1 détecte des bords et le remplissage. La couche 3 combine en coins et côtés. À la couche 6, en 8×8, le carré n'est plus visible, mais les cartes s'allument autour de la case du centre. La tête répond : une seule case dit « il y a un objet », et c'est là que « carré » est le plus probable.

**Transición (clave):** « Mais pour que le réseau apprenne ça, il faut lui dire ce qu'il aurait dû répondre : c'est la cible. »

---

## 7. La réponse attendue (5:00 – 5:50)

**Qué muestra:** un ejemplo concreto, paso a paso (círculos 1-2-3).
1. La etiqueta del archivo, `25 14 16 16 1`, significa: centro en x = 25, y = 14; ancho 16, alto 16; forma 1 = carré.
2. ¿En qué casilla cae el centro? 25 ÷ 8 = 3,1 da la columna 3, y 14 ÷ 8 = 1,75 da la fila 1. Es la casilla naranja de la imagen.
3. Esa casilla debe responder con 4 cosas (la tabla):
   - **objet ? = 1**: sí, hay una figura;
   - **dx, dy**: la parte decimal de 3,125 y 1,75, es decir, *dónde* está el centro dentro de la casilla (0 = borde izquierdo o superior, 1 = borde derecho o inferior);
   - **taille**: 16 ÷ 64 = 0,25, el tamaño como fracción de la imagen;
   - **forme = 1**: lo único nuevo respecto al TP.
- Las otras 63 casillas deben responder «objet ? = 0».

Idea a transmitir: **una sola casilla es responsable de la figura**, y le pido 4 respuestas.

**Qué decir:**
> Pour apprendre, le réseau doit savoir ce qu'il aurait dû répondre. Prenons ce carré : son étiquette dit centre en 25, 14, taille 16, forme numéro 1, carré. Je découpe l'image en 8×8 cases de 8 pixels. Le centre tombe dans la case orange : colonne 3, ligne 1. C'est elle qui doit répondre : oui, il y a un objet ; le centre est à tel endroit dans ma case ; la taille est un quart de l'image ; et c'est un carré. Les premières valeurs existaient déjà au TP, la forme est la nouveauté. Toutes les autres cases doivent dire « pas d'objet ».

**Transición (clave):** « Maintenant il faut mesurer l'écart entre ce que le réseau répond et cette réponse attendue : c'est la perte. »

---

## 8. La perte (5:50 – 7:00)

**Qué muestra:**
- **Arriba, la fórmula:** la pérdida total es la suma de 3 errores, **uno por cada pregunta** de la diapo 7. Esta es la conexión que tienes que hacer en voz alta.
- **Las 3 tarjetas:**
  - **¿Hay objeto?** Es un sí/no, así que se usa BCE (entropía cruzada binaria), igual que en el TP. Se calcula en las 64 casillas. Peso 10 en la casilla buena: como hay 1 buena por 63 vacías, sin ese peso el réseau aprendería a decir siempre «no».
  - **¿Dónde y qué tamaño?** Son números, así que se usa MSE (error al cuadrado), igual que en el TP. Solo cuenta en la casilla de la figura, porque en las demás no hay caja. Se multiplica por 10 porque estos errores son números pequeños y si no apenas contarían.
  - **¿Qué forma? (nuevo)** El réseau da una probabilidad a cada forma y la CE (entropía cruzada) mira **la probabilidad de la forma correcta**.
- **La curva:** el eje x es la probabilidad dada a la forma correcta y el eje y el error. 99 % da un error de 0,01; 33 % (al azar) da 1,1; 5 % (seguro y equivocado) da 3.
- **Frase de abajo:** ¿por qué no usar MSE sobre el número de forma? Con MSE, confundir círculo (0) con triángulo (2) costaría más que confundirlo con cuadrado (1), y eso no tiene sentido: los números de clase son solo etiquetas.

**Qué decir:**
> La perte est un seul nombre que l'entraînement fait baisser. Le réseau répond à trois questions, donc la perte a trois termes.
> Un : y a-t-il un objet ? C'est un oui/non, comme au TP, j'utilise l'entropie croisée binaire. Comme il y a une seule bonne case pour 63 vides, je donne un poids 10 à la bonne case, sinon le réseau apprendrait à toujours dire non.
> Deux : où et quelle taille ? Ce sont des nombres, comme au TP, je mesure l'écart au carré.
> Trois, la nouveauté : quelle forme ? Le réseau donne une probabilité à chaque forme, et l'entropie croisée regarde la probabilité de la bonne. Sur la courbe : à 99 %, l'erreur est presque nulle ; à 33 %, c'est-à-dire au hasard, elle vaut 1,1 ; à 5 %, elle vaut 3.
> Je n'ai pas utilisé l'écart au carré sur le numéro de forme, car une forme n'est pas une quantité : se tromper entre cercle et triangle n'est pas plus grave qu'entre cercle et carré.

**Transición:** « Une fois le réseau entraîné avec cette perte, voyons les résultats. »

---

## 9. Résultats (7:00 – 8:00)

**Qué muestra:**
- **Las 4 cifras grandes:**
  - IoU medio de 0,948;
  - 100 % de cajas correctas, es decir, con IoU > 0,5, el umbral habitual;
  - 100 % de formas correctas;
  - IoU de 0,93 en las figuras pequeñas y 0,96 en las grandes.
- **Recuadro «IoU»:** la IoU mide cuánto se solapan la caja predicha y la real. Es el área común dividida por el área total que cubren las dos. 1 = idénticas, 0 = sin contacto. 0,95 equivale a menos de un píxel de error.
- **Imagen izquierda, la matriz de confusión:** cada fila es la forma real y cada columna la forma predicha. Los números de la diagonal son los aciertos (183 círculos, 209 cuadrados, 208 triángulos). Fuera de la diagonal todo es 0: ninguna confusión.
- **Imagen derecha, el histograma de IoU:** cuántas imágenes caen en cada valor de IoU. La línea roja es el umbral 0,5. Todas las barras están a la derecha, casi todas entre 0,9 y 1.

**Qué decir:**
> Je teste le réseau sur 600 images qu'il n'a jamais vues, avec deux questions : a-t-il trouvé la bonne forme, et la boîte est-elle bien placée ?
> Pour la forme, regardez la matrice de confusion : chaque ligne est la vraie forme, chaque colonne la forme prédite. Tout est sur la diagonale, donc aucune confusion : 100 %.
> Pour la boîte, j'utilise l'IoU : la surface commune entre la boîte prédite et la vraie, divisée par leur surface totale ; 1 veut dire parfait. L'histogramme montre que toutes les images sont au-dessus de 0,5, presque toutes au-dessus de 0,9. En moyenne 0,948, soit moins d'un pixel d'écart. Les petites figures sont un peu moins précises, 0,93, car un pixel d'erreur y pèse plus.

**Transición (clave):** « Reste une question : au TP l'IoU était de 0,78, ici 0,95. Qu'est-ce qui a fait la différence ? »

---

## 10. Qu'est-ce qui fait la différence ? (8:00 – 8:50)

**Qué muestra:**
- **La tabla:** 3 modelos entrenados con las mismas imágenes y cambiando **una sola cosa** cada vez:
  - **A**: el réseau del TP, que solo busca la caja;
  - **B**: el mismo réseau, al que se le añade la forma;
  - **C**: B con el réseau más profundo, el de la diapo 5.
- **Las curvas:** la IoU (izquierda) y el porcentaje de forma correcta (derecha) a lo largo de las épocas, para cada modelo.
- **Las 3 conclusiones:**
  - de A a B la IoU no cambia (0,780 frente a 0,782): añadir la forma no perjudica la caja;
  - B ya reconoce el 100 % de las formas: la forma es fácil;
  - de B a C la IoU sube de 0,78 a 0,95: la mejora viene de la profundidad, es decir, del campo de visión más amplio.

Es un experimento de «cambiar una variable cada vez»: así sabes qué produce cada efecto.

**Qué decir:**
> Pour le savoir, j'ai entraîné trois modèles sur les mêmes images, en changeant une seule chose à la fois. A, c'est le réseau du TP qui ne cherche que la boîte. B, c'est le même réseau à qui j'ajoute la forme. C, c'est B avec le réseau plus profond.
> De A à B, l'IoU ne bouge pas : apprendre la forme ne gêne pas la localisation, et la forme est facile, B fait déjà 100 %. De B à C, l'IoU passe de 0,78 à 0,95 : c'est la profondeur, donc le champ de vision plus large, qui améliore la boîte. C'est exactement ce qu'on attendait de la diapo sur le réseau.

**Transición:** « Pour conclure… »

---

## 11. Conclusion (8:50 – 9:30)

**Qué decir:**
> Trois changements suffisent pour passer du TP à l'option B. Grâce à la couleur aléatoire, le réseau apprend vraiment la forme. Et c'est le réseau plus profond qui rend la boîte précise. Les limites : une seule figure par image, pas de rotation ni de bruit, des figures aussi larges que hautes ; sur de vraies photos ces scores ne tiendraient pas. Les suites naturelles : plusieurs figures par image, c'est l'option C, et les ancres, l'option A. Merci, je passe à la démonstration du notebook.

---

## Preguntas probables del jurado (respuestas cortas)

- **«¿Por qué 100 %? ¿No es sospechoso?»**
  > Les images sont synthétiques et propres : pas de bruit, pas de rotation, une seule figure. Les trois formes sont très différentes. Sur des images réelles ce ne serait pas le cas.
- **«¿Por qué el peso 10 en la BCE?»**
  > 1 case positive pour 63 négatives ; sans poids, répondre toujours « non » donne déjà une perte faible.
- **«¿Por qué softmax y no sigmoid para la forme?»**
  > Une seule forme par figure : les 3 probabilités doivent faire 1, c'est ce que fait le softmax.
- **«¿Qué es la NMS?»**
  > Si plusieurs cases voisines disent « objet », je garde la plus sûre et je supprime celles qui se chevauchent trop avec elle. Avec une seule figure, elle sert peu.
- **«¿Por qué no más profundo todavía?»**
  > À 43 px la case voit déjà toute la figure ; plus de couches coûterait du temps pour un gain faible.
