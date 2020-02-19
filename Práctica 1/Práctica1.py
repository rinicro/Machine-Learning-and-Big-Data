'''
Práctica 1

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def desc_grad(x, y, alpha=0.01, theta=[0,0], it=1500):
    X = np.array([np.ones(len(x)),x])
    y = np.array(y)
    thetas = [theta]
    for i in range(it):
        h = np.dot(theta,X)
        sumandos = (h - y) * X
        theta = theta - (alpha / len(x)) * np.sum(sumandos,1)
        thetas.append(theta)
    
    
    return theta

data = pd.read_csv("ex1data1.csv", names=['población', 'beneficios'])
theta = desc_grad(data['población'].tolist(), data['beneficios'].tolist())

part = np.linspace(min(data['población']),max(data['población']),1000)

plt.figure(figsize=(10,10))
plt.plot(data['población'], data['beneficios'], 'r.', label="Datos suministrados")
plt.plot(part, theta[0]+theta[1]*part, 'b', label="Recta de regresión")
plt.title("Resultado del ejercicio 1")
plt.xlabel("Población de la ciudad en decenas de millar")
plt.ylabel("Ingresos en decenas de millar")
plt.legend(loc="lower right")
plt.savefig("lin1.png")
plt.show()

