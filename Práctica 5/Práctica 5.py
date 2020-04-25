'''
Práctica 5

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


os.chdir("./resources")


# Función de coste para la regresión lineal regularizada
def coste(theta, X, Y, reg=0):
    return np.sum((np.dot(X, theta) - Y)**2) / (2*len(Y)) + reg / (2*len(Y)) * np.sum(theta[1:]**2)
    
# Gradiente de la función de coste para la regresión lineal 
# regularizada
def gradiente(theta, X, Y, reg=0):
    return np.dot((np.dot(X, theta) - Y).T, X) / len(Y) + reg / len(Y) * np.concatenate(([0], theta[1:]))

# Dada una matriz columna X, devuelve la matriz que en la columna i 
# tiene X^i, con i en [1,p]
def potencia(X, p):
    res = X
    pot = X
    for i in range(2,p+1):
        pot = pot * X
        res = np.hstack((res, pot))
    return res

# Dada una matriz de datos, los normaliza por columnas con la media
# y desviación típica dadas o, si son vacíos, para que tengan 
# media 0 y desviación típica 1, y devuelve los datos normalizados,
# la media y la desviación típica 
def normalizar(X, mu=[], sigma=[]):
    if len(mu)==0 or len(sigma)==0:
        mu = np.mean(X, 0)
        sigma = np.std(X, 0)
    X2 = (X - mu) / sigma
    return X2, mu, sigma


## Parte 1: Regresión lineal

# Leemos los datos
data = loadmat('ex5data1.mat')
y = data['y'].ravel()
yval = data['yval'].ravel()
ytest = data['ytest'].ravel()
X = data['X']
Xval = data['Xval']
Xtest = data['Xtest']
X2 = np.hstack((np.array([np.ones(len(y))]).T, X))
Xval2 = np.hstack((np.array([np.ones(len(yval))]).T, Xval))
Xtest2 = np.hstack((np.array([np.ones(len(ytest))]).T, Xtest))

# Entrenamos la regresión
theta0 = np.array([0,0])
reg = 0
theta = minimize(fun=coste, x0=theta0, args=(X2, y, reg), method='TNC', jac=gradiente)['x']

# Representamos los datos y la recta obtenida
part = np.linspace(min(X),max(X),1000)
plt.figure(figsize=(10,10))
plt.plot(X, y, 'ro', label="Datos suministrados")
plt.plot(part,theta[0]+theta[1]*part,'b',label="Recta de regresión")
plt.title("Recta de regresión obtenida")
plt.xlabel("Cambio en el nivel del agua")
plt.ylabel("Agua que sale de la presa")
plt.legend(loc="lower right")
plt.savefig("regr1.png")
plt.show()


## Parte 2: Curvas de aprendizaje para la regresión lineal

# Calculamos el error con el conjunto de entrenamiento y el de
# validación para subconjuntos crecientes de entrenamiento.
# Para calcular el error usamos la función de coste sin regularizar
errorTrain = []
errorVal = []
for i in range(1,len(X2)+1):
    theta = minimize(fun=coste, x0=theta0, args=(X2[:i],y[:i],reg), method='TNC', jac=gradiente)['x']
    errorTrain.append(coste(theta, X2[:i], y[:i]))
    errorVal.append(coste(theta, Xval2, yval))
    
# Representamos las curvas de aprendizaje obtenidas
plt.figure(figsize=(10,10))
plt.plot(range(1,len(errorTrain)+1),errorTrain,'r',label="Entrenamiento")
plt.plot(range(1,len(errorVal)+1),errorVal,'b',label="Validación")
plt.title("Curvas de aprendizaje para regresión lineal")
plt.xlabel("Número de ejemplos de entrenamiento")
plt.ylabel("Error")
plt.legend(loc="upper right")
plt.savefig("curvas1.png")
plt.show()


## Parte 3: Regresión polinomial y curvas de aprendizaje

# Normalizamos y preparamos los datos
X2, mu, sigma = normalizar(potencia(X, 8))
X2 = np.hstack((np.array([np.ones(len(y))]).T, X2))
Xval2 = normalizar(potencia(Xval, 8), mu, sigma)[0]
Xval2 = np.hstack((np.array([np.ones(len(yval))]).T, Xval2))
Xtest2 = normalizar(potencia(Xtest, 8), mu, sigma)[0]
Xtest2 = np.hstack((np.array([np.ones(len(yval))]).T, Xtest2))

# Hacemos el proceso para términos de regularización 0, 1 y 100
for reg in [0,1,100]:
    
    # Entrenamos la regresión
    theta0 = np.zeros(np.shape(X2)[1])
    theta = minimize(fun=coste, x0=theta0, args=(X2,y,reg), method='TNC', jac=gradiente)['x']
    
    # Representamos los datos y la curva obtenida
    partX = np.arange(min(X),max(X),0.05)
    partX2 = normalizar(potencia(np.array([partX]).T,8),mu,sigma)[0]
    partX2 = np.hstack((np.array([np.ones(len(partX2))]).T,partX2))
    partY = np.dot(partX2, theta)
    plt.figure(figsize=(10,10))
    plt.plot(X, y, 'ro', label="Datos suministrados")
    plt.plot(partX, partY, 'b', label="Curva de regresión")
    plt.title("Curva de regresión obtenida")
    plt.xlabel("Cambio en el nivel del agua")
    plt.ylabel("Agua que sale de la presa")
    plt.legend(loc="upper left")
    plt.savefig("regr2R"+str(reg)+".png")
    plt.show()
    
    # Calculamos el error con el conjunto de entrenamiento y el de
    # validación para subconjuntos crecientes de entrenamiento.
    # Para calcular el error usamos la función de coste sin 
    # término de regularización
    errorTrain = []
    errorVal = []
    for i in range(1,len(X2)+1):
        theta = minimize(fun=coste,x0=theta0,args=(X2[:i],y[:i],reg),method='TNC',jac=gradiente)['x']
        errorTrain.append(coste(theta, X2[:i], y[:i]))
        errorVal.append(coste(theta, Xval2, yval))
        
    # Representamos las curvas de aprendizaje obtenidas
    plt.figure(figsize=(10,10))
    plt.plot(range(1,len(errorTrain)+1), errorTrain, 'r', label="Entrenamiento")
    plt.plot(range(1,len(errorVal)+1), errorVal, 'b', label="Validación")
    plt.title("Curvas de aprendizaje para regresión polinomial")
    plt.xlabel("Número de ejemplos de entrenamiento")
    plt.ylabel("Error")
    plt.legend(loc="upper right")
    plt.savefig("curvas2"+str(reg)+".png")
    plt.show()
    

## Parte 4: Selección del parámetro de regularización

# Entrenamos la regresión para distintos valores del parámetro de 
# regularización y calculamos el error con cada uno
errorTrain = []
errorVal = []
regValues = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
for reg in regValues:
    
    # Entrenamos la regresión
    theta0 = np.zeros(np.shape(X2)[1])
    theta = minimize(fun=coste, x0=theta0, args=(X2, y, reg), method='TNC', jac=gradiente)['x']
    
    # Calculamos y almacenamos el error cometido con los datos de
    # entrenamiento y los de validación
    errorTrain.append(coste(theta, X2[:i], y[:i]))
    errorVal.append(coste(theta, Xval2, yval))
    
# Representamos el error obtenido según el valor del
# parámetro de regularización
plt.figure(figsize=(10,10))
plt.plot(regValues, errorTrain, 'r', label="Entrenamiento")
plt.plot(regValues, errorVal, 'b', label="Validación")
plt.title(r"Error según $\lambda$")
plt.xlabel(r"Valor de $\lambda$")
plt.ylabel("Error")
plt.legend(loc="upper right")
plt.savefig("errorReg.png")
plt.show()

# Tomamos el parámetro de regularización que reduce la diferencia
# entre el error con los datos de entrenamiento y el error con 
# los datos de validación
reg = regValues[np.argmin(abs(np.array(errorTrain) - np.array(errorVal)))]
print("El mejor parámetro de regularización es", reg)

# Entrenamos la regresión con dicho parámetro de regularización y 
# comprobamos el error con los datos de test
theta0 = np.zeros(np.shape(X2)[1])
theta = minimize(fun=coste, x0=theta0, args=(X2, y, reg), method='TNC', jac=gradiente)['x']
error = coste(theta, Xtest2, ytest)
print("El error con los datos de test es", error)