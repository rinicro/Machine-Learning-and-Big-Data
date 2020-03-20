'''
Práctica 3

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import fmin_tnc

os.chdir("./resources")


# Función sigmoide
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# Función de coste regularizada
def coste(theta, X, Y, lmb=1):
    gXTheta = sigmoide(np.dot(X, theta))
    factor = np.dot(np.log(gXTheta).T, Y) + np.dot(np.log(1 - gXTheta).T, 1-Y)
    return -1 / len(Y) * factor + lmb / (2 * len(Y)) * np.sum(theta**2)

# Gradiente de la función de coste regularizada
def gradiente(theta, X, Y, lmb=1):
    gXTheta = sigmoide(np.dot(X, theta))
    thetaJ = np.concatenate(([0], theta[1:]))
    return 1 / len(Y) * np.dot(X.T, gXTheta-Y) + lmb / len(Y) * thetaJ

# Entrena varios clasificadores por regresión logística con el término
# de regularización dado en 'reg' y devuelve el resultado en una matriz
# con el clasificador de la etiqueta i-ésima en dicha fila
def oneVsAll(X, y, num_et, reg):
    theta0 = np.zeros(np.shape(X)[1])
    
    Y = ((1 == y[:]) * 1)
    result = np.array([fmin_tnc(func=coste, x0=theta0, fprime=gradiente, args=(X, Y, reg))[0]])
    for i in range(2, num_et+1):
        Y = ((i == y[:]) * 1)
        result = np.vstack((result, np.array([fmin_tnc(func=coste, x0=theta0, fprime=gradiente, args=(X, Y, reg))[0]])))
        
    return result

# Dada la entrada 'X' y los pesos 'theta' de una capa de una red neuronal,
# aplica los pesos y devuelve la salida de la capa
def applyLayer(X, theta):
    thetaX = np.dot(X, theta.T)
    return sigmoide(thetaX)

# Calcula el porcentaje de acierto obtenido con la entrada dada
# y las etiquetas correctas en y
def acierto(X, Y):
    resultados = X.argmax(axis=1) + 1
    return np.count_nonzero(resultados == Y.ravel()) / len(Y)
    

## Parte 1

# Leemos los datos
data = loadmat('ex3data1.mat')
y = data['y'].ravel()
X = data['X']

# Entrenamos el clasificador con distintos valores para el término de 
# regularización y almacenamos el porcentaje de acierto para cada uno
X2 = np.hstack((np.array([np.ones(len(y))]).T,X))
porc_ac = []
for lmb in range(-10, 3):
    clas = oneVsAll(X2, y, 10, 10**lmb)
    result = sigmoide(np.dot(X2, clas.T))
    porc_ac.append(acierto(result, y)*100)
    print("Con lambda = 10^" + str(lmb) + ", el porcentaje de aciertos del clasificador es un " + str(porc_ac[-1]) + "%.")
    
# Representamos el porcentaje de aciertos del clasificador en función
# del término de regularización
plt.figure(figsize=(10,10))
plt.plot(range(-10,3), porc_ac, 'ko-')
plt.title(r"Porcentaje de aciertos según el valor de $\lambda$")
plt.xlabel(r"$\lambda = 10^{x}$")
plt.ylabel("Porcentaje de acierto")
plt.savefig("AcXLambda.png")
plt.show()


## Parte 2

# Leemos los datos
weights = loadmat('ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

# Aplicamos las dos capas de la red neuronal
lay1 = applyLayer(X2, theta1)
lay1 = np.hstack((np.array([np.ones(np.shape(lay1)[0])]).T,lay1))
lay2 = applyLayer(lay1, theta2)

# Comprobamos el porcentaje de aciertos
acR = acierto(lay2, y)
print("El porcentaje de aciertos de la red neuronal es un " + str(acR*100) + "%.")