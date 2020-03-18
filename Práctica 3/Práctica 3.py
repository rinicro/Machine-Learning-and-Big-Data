'''
Práctica 3

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
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

# Calcula el porcentaje de acierto obtenido con el valor de theta dado
# y las etiquetas correctas en y
def acierto(X, Y, theta):
    gXTheta = sigmoide(np.dot(X, theta.T))
    resultados = gXTheta.argmax(axis=1) + 1
    return np.count_nonzero(resultados == Y.ravel()) / len(Y)


## Parte 1

# Leemos los datos
data = loadmat('ex3data1.mat')
y = data['y'].ravel()
X = data['X']

# Entrenamos el clasificador
X2 = np.hstack((np.array([np.ones(len(y))]).T,X))
clas = oneVsAll(X2, y, 10, 0.1)

# Comprobamos el porcentaje de aciertos
ac = acierto(X2, y, clas)
print("El porcentaje de aciertos del clasificador es un " + str(ac*100) + "%.")