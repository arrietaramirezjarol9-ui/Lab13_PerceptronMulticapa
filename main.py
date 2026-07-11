import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Datos XOR
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,1,1,0])

modelo = MLPClassifier(
    hidden_layer_sizes=(8,),   # <-- aquí va la coma
    max_iter=1000,
    random_state=1
)

modelo.fit(X, y)

pred = modelo.predict(X)

print("Predicciones:", pred)
print("Exactitud:", accuracy_score(y, pred))