import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Definición del dataset XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])

# Crear directorio de imágenes si no existe
os.makedirs("images", exist_ok=True)

# Lista de configuraciones a probar
configuraciones = [
    {
        "id": 1,
        "nombre": "Configuración Base (4 neuronas)",
        "hidden_layer_sizes": (4,),
        "max_iter": 1000,
        "activation": "relu",
        "random_state": 1
    },
    {
        "id": 2,
        "nombre": "Configuración Insuficiente (1 neurona)",
        "hidden_layer_sizes": (1,),
        "max_iter": 1000,
        "activation": "relu",
        "random_state": 1
    },
    {
        "id": 3,
        "nombre": "Configuración Mínima (2 neuronas)",
        "hidden_layer_sizes": (2,),
        "max_iter": 1000,
        "activation": "relu",
        "random_state": 1
    },
    {
        "id": 4,
        "nombre": "Configuración Multicapa Compleja (8x4 neuronas)",
        "hidden_layer_sizes": (8, 4),
        "max_iter": 1000,
        "activation": "relu",
        "random_state": 1
    }
]

# Estructura para almacenar los resultados
resultados = []

print("=====================================================================")
print("  EVALUACIÓN DE CONFIGURACIONES DE PERCEPTRÓN MULTICAPA (MLP)")
print("=====================================================================\n")

# Iterar sobre cada configuración
for config in configuraciones:
    print(f"Entrenando: {config['nombre']}...")
    
    # Crear el modelo MLP con la configuración actual
    modelo = MLPClassifier(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        max_iter=config["max_iter"],
        activation=config["activation"],
        random_state=config["random_state"]
    )
    
    # Entrenar el modelo
    modelo.fit(X, y)
    
    # Predecir
    pred = modelo.predict(X)
    
    # Calcular exactitud
    exactitud = accuracy_score(y, pred)
    
    # Registrar resultados
    resultados.append({
        "id": config["id"],
        "nombre": config["nombre"],
        "hidden": str(config["hidden_layer_sizes"]),
        "iter": config["max_iter"],
        "pred": str(pred),
        "exactitud": exactitud,
        "iter_real": modelo.n_iter_
    })
    
    print(f"  -> Predicciones: {pred}")
    print(f"  -> Exactitud: {exactitud:.2f} (iteraciones reales: {modelo.n_iter_})\n")
    
    # Graficar la frontera de decisión
    plt.figure(figsize=(6, 5))
    
    # Crear una malla de puntos para graficar la frontera
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Graficar contornos
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    # Graficar puntos de entrenamiento
    scatter_0 = plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', s=150, linewidths=3, label='Clase 0')
    scatter_1 = plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', s=150, edgecolors='black', label='Clase 1')
    
    # Títulos y formato
    plt.title(f"{config['nombre']}\nExactitud: {exactitud:.1%} | Capas: {config['hidden_layer_sizes']}", fontsize=11, fontweight='bold')
    plt.xlabel('Característica X1', fontsize=10)
    plt.ylabel('Característica X2', fontsize=10)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    
    # Guardar gráfico
    filename = f"images/decision_boundary_config_{config['id']}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# Imprimir tabla final resumida en consola
print("\n" + "="*80)
print(f"{'ID':<3} | {'Configuración':<40} | {'Capas':<10} | {'Predicciones':<12} | {'Exactitud':<9}")
print("="*80)
for r in resultados:
    print(f"{r['id']:<3} | {r['nombre']:<40} | {r['hidden']:<10} | {r['pred']:<12} | {r['exactitud']:.1%}")
print("="*80)
print("\nLos gráficos de fronteras de decisión se guardaron en la carpeta 'images/'.")
