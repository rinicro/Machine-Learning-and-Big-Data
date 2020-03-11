'''
Práctica 1

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

from scipy.optimize import check_grad

os.chdir("./resources")


# Función sigmoide
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def coste(theta, X, Y):
    gXTheta = sigmoide(np.dot(X, theta))
    factor = np.dot(np.log(gXTheta).T, Y) + np.dot(np.log(1 - gXTheta).T, 1-Y)
    return -1 / len(Y) * factor

def gradiente(theta, X, Y):
    gXTheta = sigmoide(np.dot(X, theta))
    return 1 / len(Y) * np.dot(X.T, gXTheta-Y)

## Parte 1: Regresión con una variable
 

# Leemos los datos
data = pd.read_csv("ex2data1.csv", names=['exam1', 'exam2', 'resultado'])
X = np.array([data['exam1'], data['exam2']]).T
Y = np.array(data['resultado'])

# Representamos los datos en una gráfica
adm = np.where(Y == 1)
rech = np.where(Y == 0)
plt.figure(figsize=(10,10))
plt.scatter(X[adm, 0], X[adm, 1], marker='o', c='blue', label="Admitidos")
plt.scatter(X[rech, 0], X[rech, 1], marker='x', c='red', label="Rechazados")
plt.title("Datos proporcionados para la parte 1")
plt.xlabel("Examen 1")
plt.ylabel("Examen 2")
plt.legend(loc="upper right")
plt.savefig("p1datos.png")
plt.show()

# Calculamos el valor óptimo de theta
X2 = np.hstack((np.array([np.ones(len(Y))]).T,X))
theta0 = np.zeros(np.shape(X2)[1])
result = opt.fmin_tnc(func=coste, x0=theta0, fprime=gradiente, args=(X2, Y))
a=coste(result[0],X2,Y)


