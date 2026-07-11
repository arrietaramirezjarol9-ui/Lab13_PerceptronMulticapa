# Laboratorio N.º 13: Perceptrón Multicapa (MLP) - Problema XOR

Este repositorio contiene la resolución de la sesión práctica correspondiente al **Laboratorio 13** de la experiencia curricular de **Sistemas Inteligentes** (VII Ciclo) de la Escuela Profesional de Ingeniería de Sistemas de la **Universidad César Vallejo**.

* **Estudiante:** Jermain Jarol Arrieta Ramirez
* **Semestre:** 2026-I
* **Tema:** Perceptrón Multicapa (MLP) – Análisis del comportamiento del modelo en el problema XOR

---

## 📌 Descripción del Proyecto

El objetivo de esta práctica es implementar y analizar un modelo de **Perceptrón Multicapa (MLP)** para resolver el problema clásico de la compuerta lógica **XOR** (Exclusive OR). El problema XOR es fundamental en el estudio de las redes neuronales artificiales, ya que al no ser linealmente separable, requiere la introducción de capas ocultas con funciones de activación no lineales para poder clasificarse correctamente.

Utilizando la librería `scikit-learn` en Python, se implementa un clasificador MLP (`MLPClassifier`) y se evalúa bajo 4 configuraciones distintas de red neuronal, analizando el impacto de su topología (número de capas y neuronas ocultas) en las fronteras de decisión y en el proceso de entrenamiento.

---

## 📂 Estructura del Repositorio

El repositorio se organiza de la siguiente manera:

```text
Lab13_PerceptronMulticapa/
├── images/                   # Gráficos generados de fronteras de decisión
│   ├── decision_boundary_config_1.png
│   ├── decision_boundary_config_2.png
│   ├── decision_boundary_config_3.png
│   └── decision_boundary_config_4.png
├── informe_laboratorio.md    # Reporte detallado y cuestionario teórico resuelto
├── main.py                   # Script de prueba simple con configuración básica
├── mlp_xor.py                # Script principal de simulación y graficado
└── README.md                 # Presentación del proyecto (este archivo)
```

---

## 🛠️ Requisitos e Instalación

Para ejecutar los scripts de este repositorio localmente, asegúrate de tener instalado Python 3.10 o superior y las siguientes librerías de Python:

```bash
pip install numpy scikit-learn matplotlib
```

---

## 🚀 Ejecución

Puedes ejecutar cualquiera de las simulaciones directamente desde tu terminal:

### Simulación de prueba rápida
Ejecuta una única configuración base con 8 neuronas ocultas:
```bash
python main.py
```

### Ejecución de los Experimentos Completos y Gráficos
Ejecuta el script principal que entrena las 4 configuraciones neuronales, imprime un resumen comparativo en la consola y genera/guarda los gráficos de fronteras de decisión en la carpeta `images/`:
```bash
python mlp_xor.py
```

---

## 📊 Resultados y Experimentos

A continuación se resumen los resultados obtenidos al entrenar el modelo utilizando el optimizador estocástico Adam (`solver='adam'`), la semilla pseudoaleatoria por defecto (`random_state=1`) y un límite máximo de 1000 iteraciones:

### Tabla Comparativa de Configuraciones

| ID | Configuración | Estructura Capas | Predicciones Obtenidas | Iteraciones Reales | Exactitud (Accuracy) | Estado de Clasificación |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Configuración Base** | `(4,)` | `[0, 1, 1, 0]` | 1000 | **100.0%** | **Éxito Total** (Separación óptima) |
| **2** | **Configuración Insuficiente** | `(1,)` | `[0, 0, 0, 0]` | 754 | **50.0%** | **Fallo** (Clasifica todo como 0) |
| **3** | **Configuración Mínima** | `(2,)` | `[0, 0, 0, 0]` | 12 | **50.0%** | **Fallo** (Estancamiento por mala inicialización) |
| **4** | **Configuración Multicapa** | `(8, 4)` | `[0, 1, 0, 0]` | 826 | **75.0%** | **Fallo Parcial** (Mínimo local) |

---

## 🎨 Fronteras de Decisión Visualizadas

Las fronteras de decisión ilustran cómo las redes neuronales segmentan el plano bidimensional para separar las clases XOR (Clase 0 en rojo, Clase 1 en azul):

### 1. Configuración Base `(4,)` - Exactitud: 100%
Con 4 neuronas en la capa oculta, la red cuenta con la flexibilidad necesaria para doblar el espacio y aislar correctamente los puntos correspondientes a la compuerta XOR.
* Ver frontera: `images/decision_boundary_config_1.png`

### 2. Configuración Insuficiente `(1,)` - Exactitud: 50%
Una sola neurona oculta solo puede trazar una frontera lineal recta. Debido a que XOR no es linealmente separable, el modelo decide predecir la clase mayoritaria (0) para estabilizar el error, fallando por completo en la separación.
* Ver frontera: `images/decision_boundary_config_2.png`

### 3. Configuración Mínima `(2,)` - Exactitud: 50%
Aunque teóricamente 2 neuronas ocultas son suficientes para resolver XOR, bajo la inicialización pseudoaleatoria por defecto (`random_state=1`), los pesos cayeron en un punto de silla o gradiente plano, deteniendo el aprendizaje de manera temprana en la iteración 12.
*(Nota: Si se modifica el seed a `random_state=6`, esta configuración sí aprende correctamente el problema logrando el 100% de exactitud).*
* Ver frontera: `images/decision_boundary_config_3.png`

### 4. Configuración Multicapa `(8, 4)` - Exactitud: 75%
Con un exceso de parámetros en relación con el conjunto de datos reducido (solo 4 muestras), el modelo se volvió vulnerable a caer en mínimos locales durante la optimización con Adam, fallando en el punto `(1,0)`.
* Ver frontera: `images/decision_boundary_config_4.png`


