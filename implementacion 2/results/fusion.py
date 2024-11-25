import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Definir las matrices de confusión
matriz_1 = np.array([[236,  12,  18,  12],
                     [ 13, 202,   9,   5],
                     [ 75,  10, 228,  71],
                     [ 10,   4,  19, 161]])

matriz_2 = np.array([[273,   4,  34,  16],
                     [  7, 121,   7,   2],
                     [ 59,   9, 258,  36],
                     [  7,   1,  14, 175]])

matriz_3 = np.array([[176,  51,  31,  28],
                     [ 25, 193,  14,   8],
                     [ 37, 106, 142,  35],
                     [ 30,  49,  29, 197]])

matriz_4 = np.array([[239,  16,  45,   3],
                     [ 19, 245,  44,  19],
                     [ 56,  10, 175,  17],
                     [  5,   0,  19, 119]])

matriz_5 = np.array([[308,  35,  70,  29],
                     [  5, 145,  12,   8],
                     [ 49,  38, 268,  29],
                     [ 13,   6,  45, 181]])

# Apilar las matrices y calcular el promedio
matrices = np.array([matriz_1, matriz_2, matriz_3, matriz_4, matriz_5])
promedio_matriz = np.mean(matrices, axis=0)

# Redondear el resultado para obtener enteros
promedio_matriz_redondeado = np.round(promedio_matriz).astype(int)

# Crear la matriz de confusión combinada
plt.figure(figsize=(6, 5))
sns.heatmap(promedio_matriz_redondeado, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=['Angry', 'Happy', 'Sad', 'Neutral'], yticklabels=['Angry', 'Happy', 'Sad', 'Neutral'])

plt.title('Avg. Confusion Matrix')
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')

# Guardar la imagen
plt.savefig('/media/chega/Nuevo vol/mmer_pfc/MMER/results/matriz_confusion_promediada.png')
plt.show()




modelo 1

Matriz de confusión:
[[238  10  19  11]
 [ 14 202   9   4]
 [ 75   9 229  71]
 [ 12   4  20 158]]
F1 General del modelo: 0.7689
Recall: 0.7872

Matriz de confusión guardada en: /media/chega/Nuevo vol/mmer_pfc/MMER/results/confusion_matrix_session_1.png

Precisión por emoción:
Neutral: 0.702
Angry: 0.898
Happy: 0.827
Sad: 0.648
macro avg: 0.769
weighted avg: 0.778
The time elapse is: 00: 00: 52




modelo 2
Matriz de confusión:
[[281   3  26  17]
 [  6 121   8   2]
 [ 61   9 253  39]
 [  4   1  15 177]]
F1 General del modelo: 0.8255
Recall: 0.8350

Matriz de confusión guardada en: /media/chega/Nuevo vol/mmer_pfc/MMER/results/confusion_matrix_session_2.png

Precisión por emoción:
Neutral: 0.798
Angry: 0.903
Happy: 0.838
Sad: 0.753
macro avg: 0.823
weighted avg: 0.818
The time elapse is: 00: 00: 47

modelo 3
Matriz de confusión:
[[178  55  27  26]
 [ 26 189  17   8]
 [ 36 110 142  32]
 [ 30  51  32 192]]
F1 General del modelo: 0.6091
Recall: 0.6208

Matriz de confusión guardada en: /media/chega/Nuevo vol/mmer_pfc/MMER/results/confusion_matrix_session_3.png

Precisión por emoción:
Neutral: 0.659
Angry: 0.467
Happy: 0.651
Sad: 0.744
macro avg: 0.630
weighted avg: 0.639
The time elapse is: 00: 00: 48

modelo 4
Matriz de confusión:
[[238  14  48   3]
 [ 18 243  44  22]
 [ 57  10 172  19]
 [  3   0  22 118]]
F1 General del modelo: 0.7483
Recall: 0.7551

Matriz de confusión guardada en: /media/chega/Nuevo vol/mmer_pfc/MMER/results/confusion_matrix_session_4.png

Precisión por emoción:
Neutral: 0.753
Angry: 0.910
Happy: 0.601
Sad: 0.728
macro avg: 0.748
weighted avg: 0.762
The time elapse is: 00: 00: 44


modelo 5
Matriz de confusión:
[[313  37  67  25]
 [  6 143  11  10]
 [ 52  36 266  30]
 [ 13   6  45 181]]
F1 General del modelo: 0.7282
Recall: 0.7452

Matriz de confusión guardada en: /media/chega/Nuevo vol/mmer_pfc/MMER/results/confusion_matrix_session_5.png

Precisión por emoción:
Neutral: 0.815
Angry: 0.644
Happy: 0.684
Sad: 0.736
macro avg: 0.720
weighted avg: 0.735
The time elapse is: 00: 00: 55
