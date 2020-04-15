'''
Práctica 4

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
from scipy.optimize import minimize


os.chdir("./resources")


# Función sigmoide
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# Derivada de la función sigmoide
def diffSigmoide(a):
    return a * (1 - a)

# Devuelve una matriz de pesos aleatorios con la dimensión dada
def randomWeights(l_in, l_out):
    eps = np.sqrt(6)/np.sqrt(l_in + l_out)
    rnd = np.random.random((l_out, l_in+1)) * (2*eps) - eps
    return rnd

# Dada la entrada 'X' y los pesos 'theta' de una capa de una red 
# neuronal, aplica los pesos y devuelve la salida de la capa
def applyLayer(X, theta):
    thetaX = np.dot(X, theta.T)
    return sigmoide(thetaX)

# Dada la entrada 'X' y el array de matrices de pesos 'theta',
# devuelve la entrada de cada capa y el resultado final devuelto
# por la red neuronal
def applyNet(X, theta):
    lay = X.copy()
    a = []
    for i in range(len(theta)):
        lay = np.hstack((np.array([np.ones(len(lay))]).T, lay))
        a.append(lay.copy())
        lay = applyLayer(lay, theta[i])
    
    return lay,a

# Calcula la función de coste de una red neuronal para la 
# salida esperada 'y', el resultado de la red 'h_theta', el array
# de matrices de pesos 'theta' y el término de regularización 'reg'
def coste(y, h_theta, theta, reg):
    sumandos = -y * np.log(h_theta) - (1-y) * np.log(1-h_theta)
    regul = 0
    for i in range(len(theta)):
        regul += np.sum(theta[i][:,1:]**2)
    result = np.sum(sumandos) / len(y) + reg * regul / (2*len(y))
    return result

# Calcula el gradiente de la función de coste haciendo 
# retropropagación dada la salida esperada 'y', la entrada 
# de cada capa 'a', la salida de la red 'h_theta', el array de
# matrices de pesos 'theta' y el término de regularización 'reg'
def gradiente(y, a, h_theta, theta, reg):
    d = h_theta - y
    delta = [np.dot(d.T, a[-1]) / len(y)]
   
    for i in range(len(theta)-1,0,-1):
        d = np.dot(d, theta[i]) * diffSigmoide(a[i])
        d = d[:,1:]
        delta.insert(0, np.dot(d.T, a[i-1]) / len(y))
   
    for i in range(len(delta)):
        delta[i][:,1:] += reg * theta[i][:,1:] / len(y)
        
    return delta 

# Calcula y devuelve el coste y el gradiente de una red neuronal
# dados todos los pesos en el array 'param_rn', las dimensiones
# de cada capa en 'capas', los datos de entrada 'X', la salida
# esperada 'y' y el término de regularización 'reg'
def backprop(params_rn, capas, X, y, reg):
    # Convertimos el vector de todos los pesos en las distintas
    # matrices
    theta = [np.reshape(params_rn[:capas[1]*(capas[0]+1)],(capas[1],capas[0]+1))]
    gastados = capas[1]*(capas[0]+1)
    for i in range(len(capas)-2):
        theta.append(np.reshape(params_rn[gastados:gastados+capas[i+2]*(capas[i+1]+1)],(capas[i+2],capas[i+1]+1)))
        gastados += capas[i+2]*(capas[i+1]+1)
    
    # Calculamos el vector de salida esperada para la red neuronal
    Y = np.zeros((len(y), capas[-1]))
    for i in range(len(Y)):
        Y[i,y[i]-1] = 1
        
    # Aplicamos la red neuronal
    h_theta,a = applyNet(X, theta)
    
    cost = coste(Y, h_theta, theta, reg)
    grad = gradiente(Y, a, h_theta, theta, reg)
    
    g = np.array([])
    for i in range(len(grad)):
        g = np.concatenate((g, grad[i].ravel()))
    
    return cost, g

# Calcula el porcentaje de acierto obtenido con la respuesta dada 
# en 'X' y las etiquetas correctas en 'Y'
def acierto(X, Y):
    resultados = X.argmax(axis=1) + 1
    return 100 * np.count_nonzero(resultados == Y.ravel()) / len(Y)
    

## Parte 1: Función de coste y gradiente

# Leemos los datos
data = loadmat('ex4data1.mat')
y = data['y'].ravel()
X = data['X']

# Leemos las matrices de pesos
weights = loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']

# Calculamos el coste y el gradiente
theta = np.concatenate((theta1.ravel(), theta2.ravel()))
res = backprop(theta, (400,25,10), X, y, 1)


## Parte 2: Entrenamos la red neuronal

# Creamos unas matrices de pesos iniciales de forma aleatoria
theta01 = randomWeights(400, 25)
theta02 = randomWeights(25, 10)
theta0 = np.concatenate((theta01.ravel(), theta02.ravel()))

# Entrenamos la red neuronal con distintos valores para el
# parámetro de regularización y representamos su porcentaje
# de acierto
reg = range(-5, 3)
itera = range(10, 110, 10)

pAc = np.zeros((len(reg), len(itera)))
for i in range(len(reg)):
    for j in range(len(itera)):
        theta = minimize(fun=backprop, x0=theta0, args=((400,25,10), X, y, 10**reg[i]), method='TNC', jac=True, options={'maxiter':itera[j]})['x']

        theta1 = np.reshape(theta[:25*401],(25,401))
        theta2 = np.reshape(theta[25*401:],(10,26))
        res =  applyNet(X, (theta1, theta2))[0]

        pAc[i][j] = acierto(res, y)

colores = ['blue', 'orange', 'green', 'red', 'purple', 'pink', 'gray', 'olive', 'cyan']
plt.figure(figsize=(10,10))
for i in range(len(pAc)):
    plt.plot(range(10, 110, 10), pAc[i], color=colores[i], marker='o', label=(r"$\lambda = 10^{" + str(reg[i]) + "}$"))
plt.title(r"Porcentaje de aciertos según el valor de $\lambda$")
plt.xlabel("Iteraciones")
plt.ylabel("Porcentaje de acierto")
plt.legend(loc="lower right")
plt.savefig("LineasLambda.png")
plt.show()

xLabels = [str(it) for it in itera]
yLabels = [r'$10^{' + str(r) + '}$' for r in reg]
plt.figure(figsize=(10,10))
plt.title(r"Porcentaje de aciertos según el valor de $\lambda$")
fig = plt.subplot()
im = fig.imshow(pAc, cmap=cm.autumn_r)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(xLabels)):
    for j in range(len(yLabels)):
        text = fig.text(i, j, pAc[j][i], ha="center", va="center", color="k")
fig.set_xticks(np.arange(len(xLabels)))
fig.set_yticks(np.arange(len(yLabels)))
fig.set_xticklabels(xLabels)
fig.set_yticklabels(yLabels)
plt.savefig("CuadrosLambda.png")
plt.show()
