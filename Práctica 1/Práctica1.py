'''
Práctica 1

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def desc_grad(x, y, alpha=0.01, theta=None, it=1500):
    X = np.vstack((np.ones(len(y)), x.T)) 
    y = np.array(y)
    if (theta == None):
        theta = np.zeros(np.shape(X)[0])
    thetas = np.array([theta])
    for i in range(it):
        h = np.dot(theta, X)
        sumandos = (h - y) * X
        theta = theta - (alpha / len(y)) * np.sum(sumandos,1)
        thetas = np.append(thetas, [theta], axis=0)
    return theta, thetas, cost(thetas, X, y)

def ec_norm(x, y):
    X = np.hstack((np.array([np.ones(len(y))]).T, x))
    y = np.array(y)
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),y)
    return theta

def cost(theta, X, y):
    h = np.dot(theta, X)
    sumandos = (h - y)**2
    return 1 / (2 * len(y)) * np.sum(sumandos, 1)

def normalizar(x):
    mu = np.mean(x, 0)
    sigma = np.std(x, 0)
    x2 = (x - mu) / sigma
    return x2, mu, sigma


# Parte 1: Regresión con una variable
 

# Leemos los datos que usaremos para calcular la recta de regresión
data = pd.read_csv("ex1data1.csv", names=['población', 'beneficios'])

# Utilizamos nuestra función desc_grad para calcular los valores de theta y
# además se obtienen los valores que ha ido tomando theta durante el proceso
# y el coste para cada uno
x = np.array(data['población'].tolist()).T
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
#Theta0, Theta1 = np.meshgrid(np.arange(-10,10,0.1), np.arange(-1,4,0.1))
Theta0, Theta1 = np.meshgrid(np.arange(-4,3,0.1), np.arange(-1,6,0.1))
Costes = cost(np.transpose(np.array([np.ravel(Theta0),np.ravel(Theta1)])), np.array([np.ones(len(y)),x]), np.array(y))
Costes = np.reshape(Costes, np.shape(Theta0))
fig = plt.figure(figsize=(10,10))
plt.contour(Theta0, Theta1, Costes, np.logspace(-2,3, 20), cmap=cm.rainbow_r)
plt.plot(thetas[:,0], thetas[:,1], color='black', linewidth=2, label="Descenso de gradiente")
plt.plot(theta[0], theta[1], color='black', marker='x', label="Resultado final")
plt.title("Curvas de nivel de la función de coste de la parte 1")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.axis('equal')
plt.legend(loc="lower right")
plt.savefig("p1niv.png")
plt.show()


# Parte 2: Regresión con varias variables


# Leemos los datos que usaremos 
data = pd.read_csv("ex1data2.csv", names=['superficie', 'habitaciones', 'precio'])

# Normalizamos los datos de entrada del algoritmo
x = np.array([data['superficie'].tolist(), data['habitaciones'].tolist()]).T
normx, mu, sigma = normalizar(x)

# Utilizamos nuestra función desc_grad para calcular hacer la regresión lineal
# variando la tasa de aprendizaje. Además, vamos almacenando los valores que
# han ido tomando theta y la función de coste durante el proceso para cada 
# posible valor de alfa
# Además, representamos en una gráfica los valores que va tomando la función
# de coste a medida que avanza el descenso de gradiente para cada valor de la
# tasa de aprendizaje
y = data['precio'].tolist()
plt.figure(figsize=(10,10))
for alfa in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
    theta, thetas, costes = desc_grad(normx,y,alpha=alfa)
    plt.plot(range(len(costes)), costes, linewidth=2, label=r"$J(\theta)$ con $\alpha=$" + str(alfa))
plt.title(r"Evolución de la función de coste para distintos valores de $\alpha$")
plt.xlabel("Iteración")
plt.ylabel(r"$J(\theta)$")
plt.legend(loc="upper right")
plt.savefig("p2cost.png")
plt.show()

# Finalmente comprobamos que el resultado obtenido con el descenso de 
# gradiente es correcto comparándolo con el que se obtiene a partir de la
# ecuación normal
thetaGrad = desc_grad(normx, y)[0]
thetaNorm = ec_norm(x, y)

ex = np.array([1650, 3])
normEx = (ex-mu)/sigma
ex = np.append([1], ex)
normEx = np.append([1], normEx)

print("Usando el descenso de gradiente, la predicción es y =", np.dot(thetaGrad, normEx))
print("Usando la ecuación normal, la predicción es y =", np.dot(thetaNorm, ex))