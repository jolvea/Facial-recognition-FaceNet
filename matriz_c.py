import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial.distance import euclidean

# 1. Cargar la base de datos desde el archivo .pkl
with open("data_9.pkl", "rb") as file:
    database = pickle.load(file)

# 2. Crear listas para las etiquetas reales y predichas
true_labels = []
predicted_labels = []

# 3. Definir un umbral de similitud (por ejemplo, 0.8)
threshold = 0.8

# 4. Comparar embeddings y generar etiquetas
all_embeddings = []
all_labels = []

for person in database:
    for image in database[person]:
        all_embeddings.append(database[person][image][0])  # Embedding
        all_labels.append(person)  # Etiqueta real

# Generar etiquetas predichas basadas en la distancia mínima entre embeddings
for i, embedding in enumerate(all_embeddings):
    distances = [euclidean(embedding, ref_embedding) for ref_embedding in all_embeddings]
    distances[i] = np.inf  # Evitar comparar consigo mismo
    min_distance_index = np.argmin(distances)
    
    if distances[min_distance_index] < threshold:
        predicted_labels.append(all_labels[min_distance_index])
    else:
        predicted_labels.append("Desconocido")

    true_labels.append(all_labels[i])

# 5. Calcular la matriz de confusión
unique_labels = sorted(set(true_labels + predicted_labels))
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)

# 6. Visualizar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=unique_labels)
disp.plot(cmap="viridis", xticks_rotation="vertical")
disp.ax_.set_title("Matriz de Confusión")
disp.ax_.set_xlabel("Etiquetas Predichas")
disp.ax_.set_ylabel("Etiquetas Reales")
plt.show()











"""
A calcular:
- Precisión (Accuracy).
- Precisión Positiva (Precision).
- Sensibilidad (Recall).
- F1-Score.
"""
from sklearn.metrics import confusion_matrix, classification_report

# Generar la matriz de confusión (usando etiquetas reales y predichas del ejemplo anterior)
unique_labels = sorted(set(true_labels + predicted_labels))
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)

# Mostrar la matriz de confusión
print("Matriz de Confusión:\n", conf_matrix)

# Calcular métricas globales (Accuracy, Precision, Recall, F1-Score)
accuracy = (conf_matrix.trace()) / conf_matrix.sum()  # Traza = suma de los elementos diagonales (TP y TN)
precision = []
recall = []
f1_score = []

# Para cada clase, calcular métricas
for i, label in enumerate(unique_labels):
    tp = conf_matrix[i, i]
    fp = conf_matrix[:, i].sum() - tp
    fn = conf_matrix[i, :].sum() - tp
    tn = conf_matrix.sum() - (tp + fp + fn)
    
    class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
    
    precision.append(class_precision)
    recall.append(class_recall)
    f1_score.append(class_f1)

# Mostrar resultados generales
print(f"Accuracy: {accuracy:.2%}")
print("\nMétricas por Clase:")
for i, label in enumerate(unique_labels):
    print(f"Clase {label}:")
    print(f"  Precision: {precision[i]:.2%}")
    print(f"  Recall: {recall[i]:.2%}")
    print(f"  F1-Score: {f1_score[i]:.2%}")

# (Opcional) Usar classification_report para resumen
print("\nReporte de Clasificación (Sklearn):")
print(classification_report(true_labels, predicted_labels, labels=unique_labels))

