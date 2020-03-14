'''
Práctica 2

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from sklearn.preprocessing import PolynomialFeatures


os.chdir("./resources")


# Función sigmoide
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def costeP1(theta, X, Y):
    gXTheta = sigmoide(np.dot(X, theta))
    factor = np.dot(np.log(gXTheta).T, Y) + np.dot(np.log(1 - gXTheta).T, 1-Y)
    return -1 / len(Y) * factor

def gradienteP1(theta, X, Y):
    gXTheta = sigmoide(np.dot(X, theta))
    return 1 / len(Y) * np.dot(X.T, gXTheta-Y)

def acierto(X, Y, theta):
    gXTheta = sigmoide(np.dot(X, theta))
    resultados = [((gXTheta >= 0.5) & (Y == 1)) | ((gXTheta < 0.5) & (Y == 0))]
    return np.count_nonzero(resultados) / len(Y)

def costeP2(theta, X, Y, lmb=1):
    gXTheta = sigmoide(np.dot(X, theta))
    factor = np.dot(np.log(gXTheta).T, Y) + np.dot(np.log(1 - gXTheta).T, 1-Y)
    return -1 / len(Y) * factor + lmb / (2 * len(Y)) * np.sum(theta**2)

def gradienteP2(theta, X, Y, lmb=1):
    gXTheta = sigmoide(np.dot(X, theta))
    thetaJ = np.concatenate(([0], theta[1:]))
    return 1 / len(Y) * np.dot(X.T, gXTheta-Y) + lmb / len(Y) * thetaJ
    

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
plt.savefig("p1Datos.png")
plt.show()

# Calculamos el valor óptimo de theta
X2 = np.hstack((np.array([np.ones(len(Y))]).T,X))
theta0 = np.zeros(np.shape(X2)[1])
result = fmin_tnc(func=costeP1, x0=theta0, fprime=gradienteP1, args=(X2, Y))[0]

# Representamos la recta en una gráfica junto a los datos
part = np.linspace(min(X[:,0]), max(X[:,0]), 1000)
plt.figure(figsize=(10,10))
plt.scatter(X[adm, 0], X[adm, 1], marker='o', c='blue', label="Admitidos")
plt.scatter(X[rech, 0], X[rech, 1], marker='x', c='red', label="Rechazados")
plt.plot(part, -(result[0]+result[1]*part)/result[2], 'k', label="Recta de decisión")
plt.title("Recta de decisión generada para la parte 1")
plt.xlabel("Examen 1")
plt.ylabel("Examen 2")
plt.legend(loc="upper right")
plt.savefig("p1Recta.png")
plt.show()

# Comprobamos el porcentaje de aciertos de la recta obtenida
porc_ac = acierto(X2, Y, result)
print("Se tiene un " + str(porc_ac*100) + "% de aciertos")


## Parte 2


# Leemos los datos
data = pd.read_csv("ex2data2.csv", names=['test1', 'test2', 'resultado'])
X = np.array([data['test1'], data['test2']]).T
Y = np.array(data['resultado'])

# Representamos los datos en una gráfica
corr = np.where(Y == 1)
defe = np.where(Y == 0)
plt.figure(figsize=(10,10))
plt.scatter(X[corr, 0], X[corr, 1], marker='o', c='blue', label="Correctos")
plt.scatter(X[defe, 0], X[defe, 1], marker='x', c='red', label="Defectuosos")
plt.title("Datos proporcionados para la parte 2")
plt.xlabel("Test 1")
plt.ylabel("Test 2")
plt.legend(loc="upper right")
plt.savefig("p2Datos.png")
plt.show()

# Aplicamos PolynomialFeatures para mapear los atributos y obtener 
# un mejor ajuste a los ejemplos de entrenamiento
X2 = PolynomialFeatures(6).fit_transform(X)

# Calculamos el valor óptimo de theta
theta0 = np.zeros(np.shape(X2)[1])
result = fmin_tnc(func=costeP2, x0=theta0, fprime=gradienteP2, args=(X2, Y))[0]

# Representamos el resultado obtenido
xx1, xx2 = np.meshgrid(np.linspace(min(X[:,0]),max(X[:,0])), np.linspace(min(X[:,1]),max(X[:,1])))
mesh = PolynomialFeatures(6).fit_transform(np.c_[xx1.ravel(),xx2.ravel()])
h = sigmoide(np.dot(mesh,result))
h = h.reshape(xx1.shape)
plt.figure(figsize=(10,10))
plt.scatter(X[corr, 0], X[corr, 1], marker='o', c='blue', label="Correctos")
plt.scatter(X[defe, 0], X[defe, 1], marker='x', c='red', label="Defectuosos")
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='k')
plt.title("Función de decisión generada para la parte 2")
plt.xlabel("Test 1")
plt.ylabel("Test 2")
plt.legend(loc="upper right")
plt.savefig("p2FuncionL1.png")
plt.show()

# Comprobamos el porcentaje de aciertos de la recta obtenida
porc_ac = acierto(X2, Y, result)
print("Se tiene un " + str(porc_ac*100) + "% de aciertos")

# Repetimos el proceso anterior con distintos valores de lambda
porc_ac = []
for lmb in range(-10, 3):
    theta0 = np.zeros(np.shape(X2)[1])
    result = fmin_tnc(func=costeP2, x0=theta0, fprime=gradienteP2, args=(X2, Y, 10**lmb))[0]
    
    xx1, xx2 = np.meshgrid(np.linspace(min(X[:,0]),max(X[:,0])), np.linspace(min(X[:,1]),max(X[:,1])))
    mesh = PolynomialFeatures(6).fit_transform(np.c_[xx1.ravel(),xx2.ravel()])
    h = sigmoide(np.dot(mesh,result))
    h = h.reshape(xx1.shape)
    plt.figure(figsize=(10,10))
    plt.scatter(X[corr, 0], X[corr, 1], marker='o', c='blue', label="Correctos")
    plt.scatter(X[defe, 0], X[defe, 1], marker='x', c='red', label="Defectuosos")
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='k')
    plt.title(r"Función de decisión generada para $\lambda = $" + str(10**lmb))
    plt.xlabel("Test 1")
    plt.ylabel("Test 2")
    plt.legend(loc="upper right")
    plt.savefig("p2FuncionVL" + str(lmb) + ".png")
    plt.show()
    
    porc_ac.append(acierto(X2, Y, result)*100)
    print("Se tiene un " + str(porc_ac[-1]) + "% de aciertos para lambda " + str(10**lmb))

# Representamos el porcentaje de aciertos según el valor de lambda
plt.figure(figsize=(10,10))
plt.plot(range(-10,3), porc_ac, 'ko-')
plt.title(r"Porcentaje de aciertos según el valor de $\lambda$")
plt.xlabel(r"$\lambda = 10^{x}$")
plt.ylabel("Porcentaje de acierto")
plt.savefig("AcXLambda.png")
plt.show()