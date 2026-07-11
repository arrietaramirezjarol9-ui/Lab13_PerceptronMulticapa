# INFORME DE LABORATORIO N.º 13: PERCEPTRÓN MULTICAPA (MLP)

**UNIVERSIDAD CÉSAR VALLEJO**  
**PROGRAMA ACADÉMICO DE INGENIERÍA DE SISTEMAS**  

* **Experiencia Curricular:** Sistemas Inteligentes  
* **Ciclo:** VII  
* **Semestre:** 2026-I  
* **Sesión N.º:** 13  
* **Integrante:** Jermain Jarol Arrieta Ramirez  
* **Tema:** Perceptrón Multicapa (MLP) – Análisis del comportamiento del modelo  

---

## 1. Introducción y Metodología

El objetivo de esta práctica es aplicar y analizar un modelo de **Perceptrón Multicapa (MLP)** para resolver el problema clásico de la compuerta lógica **XOR** (Exclusive OR). El problema XOR es históricamente significativo en la inteligencia artificial, ya que demostró las limitaciones de los perceptrones simples (lineales) y la necesidad de introducir capas ocultas con funciones de activación no lineales.

### Algoritmo e Implementación

Utilizando la librería `scikit-learn` en Python, implementamos un clasificador MLP (`MLPClassifier`) y lo evaluamos en un conjunto de datos XOR definido de la siguiente forma:

* **Entradas (X):**
  $$[0, 0], [0, 1], [1, 0], [1, 1]$$
* **Salidas Deseadas (y):**
  $$0, 1, 1, 0$$

Para un análisis detallado, se construyó un script de automatización (`mlp_xor.py`) que entrena el modelo con 4 configuraciones distintas de red neuronal, calcula la exactitud de clasificación (`accuracy_score`) y grafica la **frontera de decisión** resultante en un plano bidimensional.

> [!NOTE]
> **Aclaración sobre la Guía del Laboratorio (Histogramas):**  
> La guía original menciona en el punto 7.3 y en la metodología la *"correcta generación de histogramas y comparación de distribuciones de imágenes"*. Dado que el problema planteado es de clasificación lógica bidimensional (XOR) y no de procesamiento de imágenes, esta referencia se identifica como un error de copy-paste en la plantilla de la guía de un laboratorio previo. En su lugar, se generan y analizan las **Fronteras de Decisión**, las cuales representan el equivalente visual idóneo para observar cómo el modelo distribuye y clasifica el espacio bidimensional.

---

## 2. Resultados de las Configuraciones Experimentales

A continuación se detallan los resultados obtenidos al entrenar el MLP con el optimizador estocástico por defecto (`solver='adam'`, `random_state=1`, y un límite de 1000 iteraciones):

### Tabla Comparativa de Resultados

| ID | Configuración | Estructura Capas | Predicciones Obtenidas | Iteraciones Reales | Exactitud (Accuracy) | Estado de Clasificación |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Configuración Base (Guía)** | `(4,)` | `[0, 1, 1, 0]` | 1000 | **100.0%** | **Éxito Total** |
| **2** | **Configuración Insuficiente** | `(1,)` | `[0, 0, 0, 0]` | 754 | **50.0%** | **Fallo** (Clasifica todo como 0) |
| **3** | **Configuración Mínima** | `(2,)` | `[0, 0, 0, 0]` | 12 | **50.0%** | **Fallo** (Estancamiento temprano) |
| **4** | **Configuración Multicapa** | `(8, 4)` | `[0, 1, 0, 0]` | 826 | **75.0%** | **Fallo Parcial** (Mínimo local) |

---

## 3. Análisis Visual de Fronteras de Decisión

Las siguientes imágenes muestran cómo cada configuración del perceptrón modela el plano cartesiano para separar las clases 0 (cruces rojas) y 1 (círculos azules):

### Configuración 1: Base (4 neuronas ocultas)
* **Exactitud:** 100.0%  
* **Análisis:** Con 4 neuronas en una sola capa oculta, la red logra doblar el espacio lo suficiente para aislar las esquinas `(0,1)` y `(1,0)` (clase 1) de `(0,0)` y `(1,1)` (clase 0). Genera una frontera poligonal exitosa.
* **Frontera de decisión:** Referencia a [Boundary Config 1](images/decision_boundary_config_1.png).

### Configuración 2: Insuficiente (1 neurona oculta)
* **Exactitud:** 50.0%  
* **Análisis:** Una sola neurona oculta solo puede trazar una única línea recta en el plano. Al ser XOR no separable linealmente, el modelo es incapaz de separar ambas clases y decide clasificar todo el espacio como clase 0 para minimizar el error global, fallando en resolver el problema lógico.
* **Frontera de decisión:** Referencia a [Boundary Config 2](images/decision_boundary_config_2.png).

### Configuración 3: Mínima (2 neuronas ocultas)
* **Exactitud:** 50.0%  
* **Análisis:** Teóricamente, 2 neuronas ocultas bastan para resolver XOR. Sin embargo, bajo la semilla `random_state=1` con el optimizador Adam, el modelo se estancó de inmediato en la iteración 12 debido a una mala inicialización de pesos (saddle point o gradiente plano), lo que impidió el aprendizaje. 
* *Nota experimental:* Modificando la semilla a `random_state=6`, esta misma red de 2 neuronas alcanza el **100.0%** de exactitud.
* **Frontera de decisión:** Referencia a [Boundary Config 3](images/decision_boundary_config_3.png).

### Configuración 4: Multicapa Compleja (8x4 neuronas ocultas)
* **Exactitud:** 75.0%  
* **Análisis:** Con 12 neuronas distribuidas en dos capas, la red tiene capacidad matemática de sobra. Sin embargo, al ser un dataset tan pequeño (solo 4 muestras) y usar el solucionador estocástico Adam, la red cayó en un **mínimo local** durante el entrenamiento, clasificando erróneamente el punto `(1,0)`.
* **Frontera de decisión:** Referencia a [Boundary Config 4](images/decision_boundary_config_4.png).

---

## 4. Cuestionario Teórico del Laboratorio

### 1. ¿Por qué el perceptrón simple no puede resolver XOR?

El perceptrón simple es un clasificador lineal definido matemáticamente por:
$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)$$
Donde $f$ es una función de paso escalón. Esto implica que la frontera de decisión generada en el espacio de entrada es un **hiperplano lineal** (en 2D, una línea recta).

El problema XOR requiere la siguiente lógica de clasificación:
* Puntos Clase 0: $(0,0)$ y $(1,1)$
* Puntos Clase 1: $(0,1)$ y $(1,0)$

Si graficamos estos puntos en un plano de dos dimensiones, observamos que las clases ocupan esquinas opuestas de un cuadrado. Es geométricamente imposible trazar una sola línea recta que deje a un lado las cruces rojas y al otro los círculos azules. Dado que XOR **no es linealmente separable**, un clasificador lineal como el perceptrón simple está limitado a obtener un máximo de 75% de exactitud en el mejor de los casos (o 50% si se sesga hacia una sola clase), fallando estructuralmente.

### 2. ¿Qué papel cumplen las capas ocultas?

Las capas ocultas actúan como extractores de características intermedias. Su función principal es **transformar el espacio de representación original** de los datos de entrada en un nuevo espacio donde el problema sí sea linealmente separable.

Cada neurona de la capa oculta aplica una combinación lineal seguida de una **función de activación no lineal** (como ReLU o Sigmoide). Esto permite doblar, curvar o proyectar el espacio bidimensional de entrada a dimensiones superiores. 

Para resolver XOR con una sola capa oculta de 2 neuronas:
1. La primera neurona oculta puede aprender a clasificar la condición límite de una compuerta **OR** ($x_1 + x_2 \ge 1$).
2. La segunda neurona oculta puede aprender la condición límite de una compuerta **NAND** ($x_1 + x_2 < 1.5$).
3. La neurona de salida combina estas dos representaciones lineales intermedias mediante una operación **AND**, resolviendo la no linealidad del problema original.

### 3. ¿Qué sucede si se aumentan demasiado las neuronas?

Aumentar en exceso la cantidad de neuronas (sobreparametrización) genera varios efectos negativos:

1. **Sobreajuste (Overfitting):** El modelo adquiere tanta "capacidad de memorización" que en lugar de aprender las reglas generales de los datos, memoriza las muestras particulares del entrenamiento (incluyendo su ruido). Aunque en XOR el dataset es determinista y no tiene ruido, en problemas reales esto destruye la capacidad de generalización del modelo ante nuevos datos.
2. **Inestabilidad en la Optimización (Mínimos Locales):** En datasets sumamente pequeños como XOR, una red muy grande crea un "paisaje de pérdida" sumamente complejo con múltiples valles planos y mínimos locales. Como se observó en la Configuración 4 (8,4), los optimizadores estocásticos como Adam se confunden o se quedan atrapados con facilidad en estos mínimos locales antes de hallar la solución óptima.
3. **Costo Computacional Innecesario:** Cada neurona adicional multiplica el número de pesos de la red, aumentando los requisitos de memoria y el tiempo requerido para el paso forward y backpropagation durante el entrenamiento.

---

## 5. Conclusiones

1. **La no linealidad exige profundidad o amplitud:** Se comprobó experimentalmente que el perceptrón requiere capas ocultas y funciones de activación no lineales (como ReLU) para resolver problemas no lineales como XOR.
2. **La inicialización y optimización son críticas:** A pesar de tener la arquitectura correcta, la red de 2 neuronas falló bajo la semilla por defecto (`random_state=1`), demostrando que en redes neuronales, el éxito no solo depende de la estructura, sino también de la inicialización de los pesos y el comportamiento del optimizador.
3. **Más grande no siempre es mejor:** La configuración base de 4 neuronas resolvió el problema al 100%, mientras que una red más grande y compleja de 8x4 neuronas falló por atraparse en un mínimo local, confirmando el principio de parsimonia en aprendizaje automático (modelos más simples suelen ser más robustos).
