'''
Práctica 0

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scpint

# Función con la que trabajaremos f(x) = sin(x) + 1
def f(x):
    return (np.sin(x)+1)

# Integración por el método de Monte Carlo utilizando un bucle
def integra_mc_it(fun, a, b, num_puntos=10000):
    M = max(map(fun, np.linspace(a, b, 10000)))
    Ndeb = 0
    for i in range(num_puntos):
        x = (b-a) * np.random.rand() + a
        y = M * np.random.rand()
        if fun(x) > y:
            Ndeb += 1
    return (Ndeb / num_puntos) * (b - a) * M

# Integración por el método de Monte Carlo utilizando arrays de numpy
def integra_mc_vc(fun, a, b, num_puntos=10000):
    M = max(map(fun, np.linspace(a, b, 10000)))
    x = (b-a) * np.random.rand(1,num_puntos) + a
    y = M * np.random.rand(1,num_puntos)
    Ndeb = np.count_nonzero(fun(x) > y)
    return (Ndeb / num_puntos) * (b - a) * M

# Representamos la función que vamos a integrar
plt.figure(figsize=(10,10))
plt.plot(np.linspace(0,2*np.pi,100000),list(map(f,np.linspace(0,2*np.pi,100000))))
plt.title("Función de la que se va a calcular la integral")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.savefig("func.png")
plt.show()

# Calculamos en un bucle el tiempo de ejecución para cada una de las opciones
# y el error cometido, haciéndolo cada vez con más precisión
times = ([],[])
error = ([],[])
res = scpint.quad(f,0,2*np.pi)[0]
for n in np.linspace(1000,1000000,51):
    time1 = time.process_time()
    res1 = integra_mc_it(f,0,2*np.pi, int(n))
    time2 = time.process_time()
    res2 = integra_mc_vc(f,0,2*np.pi, int(n))
    time3 = time.process_time()
    times[0].append(time2 - time1)
    times[1].append(time3 - time2)
    error[0].append(abs(res - res1))
    error[1].append(abs(res - res2))
    
# Representamos el tiempo que tardan las dos funciones en ejecutarse según la
# precisión que utilicemos
plt.figure(figsize=(10,10))
plt.plot(np.linspace(1000,1000000,51), times[0], 'r', label="Tiempo utilizando un bucle")
plt.plot(np.linspace(1000,1000000,51), times[1], 'g', label="Tiempo utilizando un vector")
plt.title("Tiempo de ejecución de 'integra_mc'")
plt.xlabel("Número de puntos")
plt.ylabel("Tiempo de ejecución")
plt.legend(loc="upper left")
plt.savefig("times.png")
plt.show()

# Representamos el error cometido por las dos funciones según la precisión que
# utilicemos
plt.figure(figsize=(10,10))
plt.plot(np.linspace(1000,1000000,51), error[0], 'r', label="Error utilizando un bucle")
plt.plot(np.linspace(1000,1000000,51), error[1], 'g', label="Error utilizando un vector")
plt.title("Error en el resultado de 'integra_mc'")
plt.xlabel("Número de puntos")
plt.ylabel("Error cometido")
plt.legend(loc="upper right")
plt.savefig("error.png")
plt.show()