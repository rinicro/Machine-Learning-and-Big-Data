'''
Práctica 1

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def desc_grad(x, y, alpha=0.01, theta=[0,0], it=1500):
    X = np.array([np.ones(len(y)),x])
    y = np.array(y)
    thetas = np.array([theta])
    for i in range(it):
        h = np.dot(theta, X)
        sumandos = (h - y) * X
        theta = theta - (alpha / len(y)) * np.sum(sumandos,1)
        thetas = np.append(thetas, [theta], axis=0)
    return theta, thetas, cost(thetas, X, y)

def cost(theta, X, y):
    h = np.dot(theta, X)
    sumandos = (h - y)**2
    return 1 / (2 * len(y)) * np.sum(sumandos, 1)


# Parte 1: Regresión con una variable
 

# Leemos los datos que usaremos para calcular la recta de regresión
data = pd.read_csv("ex1data1.csv", names=['población', 'beneficios'])

# Utilizamos nuestra función desc_grad para calcular los valores de theta y
# además se obtienen los valores que ha ido tomando theta durante el proceso
# y el coste para cada uno
x = data['población'].tolist()
y = data['beneficios'].tolist()
theta, thetas, costes = desc_grad(x, y, theta=[0.9,3.9])

# Representamos los datos y la recta de regresión obtenida
part = np.linspace(min(data['población']),max(data['población']),1000)
plt.figure(figsize=(10,10))
plt.plot(data['población'], data['beneficios'], 'r.', label="Datos suministrados")
plt.plot(part, theta[0]+theta[1]*part, 'b', label="Recta de regresión")
plt.title("Recta de regresión lineal de la parte 1")
plt.xlabel("Población de la ciudad en decenas de millar")
plt.ylabel("Ingresos en decenas de millar")
plt.legend(loc="lower right")
plt.savefig("p1recta.png")
plt.show()

# Representamos la superficie de la función de coste según el valor de 
# theta y los valores que ha ido recorriendo nuestro descenso de gradiente
Theta0, Theta1 = np.meshgrid(np.arange(-4,1,0.1), np.arange(-1,4,0.1))
Costes = cost(np.transpose(np.array([np.ravel(Theta0),np.ravel(Theta1)])), np.array([np.ones(len(y)),x]), np.array(y))
Costes = np.reshape(Costes, np.shape(Theta0))
fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.plot_surface(Theta0, Theta1, Costes, cmap=cm.rainbow_r, linewidth=0, antialiased=False)
ax.plot_wireframe(Theta0, Theta1, Costes, rstride=5, cstride=5, color='white')
ax.plot(thetas[:,0], thetas[:,1], costes, color='black', linewidth=2, zorder=4, label="Descenso de gradiente")
ax.plot([theta[0]], [theta[1]], [costes[-1]], color='black', marker='x', zorder=5, label="Resultado final")
ax.view_init(None,-15)
plt.title("Superficie de la función de coste de la parte 1")
ax.set_xlabel(r"$\theta_0$")
ax.set_ylabel(r"$\theta_1$")
ax.set_zlabel(r"Coste $J(\theta)$")
plt.legend(loc="lower right")
plt.savefig("p1sup.png")
plt.show()

# Representamos las curvas de nivel de la función de coste según el valor
# de theta y los valores que ha ido recorriendo nuestro descenso de gradiente
Theta0, Theta1 = np.meshgrid(np.arange(-10,10,0.1), np.arange(-1,4,0.1))
Costes = cost(np.transpose(np.array([np.ravel(Theta0),np.ravel(Theta1)])), np.array([np.ones(len(y)),x]), np.array(y))
Costes = np.reshape(Costes, np.shape(Theta0))
fig = plt.figure(figsize=(10,10))
plt.contour(Theta0, Theta1, Costes, np.logspace(-2,3, 20), cmap=cm.rainbow_r)
plt.plot(thetas[:,0], thetas[:,1], color='black', linewidth=2, label="Descenso de gradiente")
plt.plot(theta[0], theta[1], color='black', marker='x', label="Resultado final")
plt.title("Curvas de nivel de la función de coste de la parte 1")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.legend(loc="lower right")
plt.savefig("p1niv.png")
plt.show()


# Parte 2: Regresión con varias variables