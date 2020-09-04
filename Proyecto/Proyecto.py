# -*- coding: utf-8 -*-
'''
Proyecto de la asignatura Aprendizaje Automático y Big Data

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from scipy.optimize import fmin_tnc, minimize
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

os.chdir("./resources")


#%% Funciones auxiliares


# Función sigmoide
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# Derivada de la función sigmoide
def diffSigmoide(a):
    return a * (1 - a)


## Funciones para regresión logística (práctica 2):
    
# Función de coste
def P1coste(theta, X, Y, reg=0):
    gXTheta = sigmoide(np.dot(X, theta))
    factor = np.dot(np.log(gXTheta).T, Y) + np.dot(np.log(1 - gXTheta).T,
                                                   1-Y)
    return -1 / len(Y) * factor + reg / (2 * len(Y)) * np.sum(theta**2)

# Gradiente de la función de coste
def P1gradiente(theta, X, Y, reg=0):
    gXTheta = sigmoide(np.dot(X, theta))
    thetaJ = np.concatenate(([0], theta[1:]))
    return 1 / len(Y) * np.dot(X.T, gXTheta-Y) + reg / len(Y) * thetaJ

# Función que devuelve el porcentaje de acierto de un resultado 
# según el valor real
def P1porc_ac(X, Y, theta):
    gXTheta = sigmoide(np.dot(X, theta))
    resultados = [((gXTheta >= 0.5) & (Y == 1)) | ((gXTheta < 0.5) 
                                                   & (Y == 0))]
    return np.count_nonzero(resultados) / len(Y) * 100


## Funciones para redes neuronales (práctica 4):

# Devuelve una matriz de pesos aleatorios con la dimensión dada
def P2randomWeights(l_in, l_out):
    eps = np.sqrt(6)/np.sqrt(l_in + l_out)
    rnd = np.random.random((l_out, l_in+1)) * (2*eps) - eps
    return rnd

# Dada la entrada 'X' y los pesos 'theta' de una capa de una red 
# neuronal, aplica los pesos y devuelve la salida de la capa
def P2applyLayer(X, theta):
    thetaX = np.dot(X, theta.T)
    return sigmoide(thetaX)

# Dada la entrada 'X' y el array de matrices de pesos 'theta',
# devuelve la entrada de cada capa y el resultado final devuelto
# por la red neuronal
def P2applyNet(X, theta):
    lay = X.copy()
    a = []
    for i in range(len(theta)):
        lay = np.hstack((np.array([np.ones(len(lay))]).T, lay))
        a.append(lay.copy())
        lay = P2applyLayer(lay, theta[i])
    
    return lay,a

# Calcula la función de coste de una red neuronal para la 
# salida esperada 'y', el resultado de la red 'h_theta', el array
# de matrices de pesos 'theta' y el término de regularización 'reg'
def P2coste(y, h_theta, theta, reg):
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
def P2gradiente(y, a, h_theta, theta, reg):
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
def P2backprop(params_rn, capas, X, Y, reg):
    # Convertimos el vector de todos los pesos en las distintas
    # matrices
    theta = [np.reshape(params_rn[:capas[1]*(capas[0]+1)],
                        (capas[1],capas[0]+1))]
    gastados = capas[1]*(capas[0]+1)
    for i in range(len(capas)-2):
        theta.append(np.reshape(params_rn[gastados:gastados+capas[i+2]*
                            (capas[i+1]+1)],(capas[i+2],capas[i+1]+1)))
        gastados += capas[i+2]*(capas[i+1]+1)
        
    # Aplicamos la red neuronal
    h_theta,a = P2applyNet(X, theta)
    
    cost = P2coste(Y, h_theta, theta, reg)
    grad = P2gradiente(Y, a, h_theta, theta, reg)
    
    g = np.array([])
    for i in range(len(grad)):
        g = np.concatenate((g, grad[i].ravel()))
    
    return cost, g

# Calcula el porcentaje de acierto dada la respuesta de la red y el
# resultado real
def P2porc_ac(res, Y):
    resultados = [((res >= 0.5) & (Y == 1)) | ((res < 0.5) & (Y == 0))]
    return np.count_nonzero(resultados) / len(Y) * 100


#%% Lectura y estudio de los datos


np.random.seed(27)

data = pd.read_csv('mushrooms.csv')

# Transformamos 
Y = data['class'].replace({'p':0, 'e':1})
X = pd.get_dummies(data.drop('class', axis=1))

# Dividimos los datos en entrenamiento, validación y test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, 
                            random_state=0, shuffle=True, stratify=Y)
Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, 
       test_size=0.25, random_state=0, shuffle=True, stratify=Ytrain)

# Preparamos los datos
Xtrain2 = np.hstack((np.array([np.ones(len(Ytrain))]).T, Xtrain))
Xval2 = np.hstack((np.array([np.ones(len(Yval))]).T, Xval))
Xtest2 = np.hstack((np.array([np.ones(len(Ytest))]).T, Xtest))
Ytrain2 = np.array([Ytrain]).T
Yval2 = np.array([Yval]).T
Ytest2 = np.array([Ytest]).T

# Representamos un histograma para cada variable según la distribución de
# champiñones venenosos y comestibles para cada posible valor
for name in data.columns[1:]:
    plt.figure(figsize=(10,10))    
    plt.title("Número de venenosos y comestibles según " + name)
    values = data[name].value_counts().axes[0].to_list()
    cuentaP = []
    cuentaE = []
    for v in values:
        cuentaP.append(len(data[(data[name]==v) & (data['class']=='p')]))
        cuentaE.append(len(data[(data[name]==v) & (data['class']=='e')]))
    plt.bar(np.arange(len(values)), cuentaP, 0.4, color='darkorchid')
    plt.bar(np.arange(len(values))+0.4, cuentaE, 0.4, color='greenyellow')
    plt.ylabel('Número de casos')
    plt.xlabel(name)
    plt.xticks(np.arange(len(values))+0.2, values)
    plt.savefig("var" + name + ".pdf", format='pdf')
    plt.show()


#%% Parte 1: Regresión logística


# Entrenamos la regresión con distintos valores para el término de 
# regularización
theta0 = np.zeros(np.shape(Xtrain2)[1])
regValues = range(-10, 4)
thetas = []
errorTrain = []
acTrain = []
errorVal = []
acVal = []
for reg in regValues:
    theta = fmin_tnc(func=P1coste, x0=theta0, fprime=P1gradiente,
                              args=(Xtrain2, Ytrain, 10**reg))[0]
    thetas.append(theta)
    errorTrain.append(P1coste(theta, Xtrain2, Ytrain))
    acTrain.append(P1porc_ac(Xtrain2, Ytrain, theta))
    errorVal.append(P1coste(theta, Xval2, Yval))
    acVal.append(P1porc_ac(Xval2, Yval, theta))
    
# Comprobamos el error y el pocentaje de acierto según el término de
# regularización
opt = np.argmin(errorVal)
print('El valor óptimo del parámetro de regularización es', 
                                              10**regValues[opt])

plt.figure(figsize=(10,10))
plt.plot(regValues, acTrain, 'r', label="Entrenamiento")
plt.plot(regValues, acVal, 'b', label="Validación")
plt.title(r"Porcentaje de acierto según $\lambda$")
plt.xlabel(r"Valor de $\lambda = 10^x$")
plt.ylabel("Porcentaje de acierto")
plt.legend(loc="lower left")
plt.savefig("aciertoLogistica.pdf", format='pdf')
plt.show()

plt.figure(figsize=(10,10))
plt.plot(regValues, errorTrain, 'r', label="Entrenamiento")
plt.plot(regValues, errorVal, 'b', label="Validación")
plt.title(r"Error según $\lambda$")
plt.xlabel(r"Valor de $\lambda = 10^x$")
plt.ylabel("Error")
plt.legend(loc="upper left")
plt.savefig("errorLogistica.pdf", format='pdf')
plt.show()
    
# Calculamos el porcentaje de acierto sobre los datos de test para el
# valor escogido del término de regularización
ac = P1porc_ac(Xtest2, Ytest, thetas[opt])
print('El porcentaje de acierto sobre los datos de test es', ac, '%')


#%% Parte 2: Redes neurales


# Creamos unas matrices inciales con pesos aleatorios
size2 = 25
theta01 = P2randomWeights(np.shape(Xtrain)[1], size2)
theta02 = P2randomWeights(size2, 1)
theta0 = np.concatenate((theta01.ravel(), theta02.ravel()))

regValues = range(-6, 4)
itera = range(10, 110, 10)

errorTrain = np.zeros((len(regValues), len(itera)))
acTrain = np.zeros((len(regValues), len(itera)))
errorVal = np.zeros((len(regValues), len(itera)))
acVal = np.zeros((len(regValues), len(itera)))

for i in range(len(regValues)):
    for j in range(len(itera)):
        theta = minimize(fun=P2backprop, x0=theta0, 
                args=((np.shape(Xtrain)[1],size2,1), Xtrain, Ytrain2,
                10**regValues[i]), method='TNC', jac=True, 
                options={'maxiter':itera[j]})['x']

        theta1 = np.reshape(theta[:size2*(np.shape(Xtrain)[1]+1)],
                            (size2,np.shape(Xtrain)[1]+1))
        theta2 = np.reshape(theta[size2*(np.shape(Xtrain)[1]+1):],
                            (1,size2+1))
        
        resTrain =  P2applyNet(Xtrain, (theta1, theta2))[0]
        acTrain[i][j] = P2porc_ac(resTrain, Ytrain2)
        resVal = P2applyNet(Xval, (theta1, theta2))[0]
        acVal[i][j] = P2porc_ac(resVal, Yval2)
        
        errorTrain[i][j] = P2coste(Ytrain2, resTrain, [theta1,theta2], 0)
        errorVal[i][j] = P2coste(Yval2, resVal, [theta1,theta2], 0)
        
# Comprobamos el error y el pocentaje de acierto según el término de
# regularización y el número de iteraciones
opt = np.argmin(errorVal)
optReg, optItera = 10**regValues[opt//len(itera)], itera[opt%len(itera)]
print('El valor óptimo del parámetro de regularización es', optReg)
print('El valor óptimo para el número de iteraciones es', optItera)

xLabels = [str(it) for it in itera]
yLabels = [r'$10^{' + str(r) + '}$' for r in regValues]

plt.figure(figsize=(10,10))
plt.title(r"Porcentaje de aciertos según el valor de $\lambda$ y" + 
          " el número de iteraciones")
plt.ylabel(r'$\lambda$')
plt.xlabel('Iteraciones')
fig = plt.subplot()
im = fig.imshow(acVal, cmap=cm.viridis)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(xLabels)):
    for j in range(len(yLabels)):
        text = fig.text(i, j, round(acVal[j][i],2), ha="center", 
               va="center", color=("k" if acVal[j][i] > 93 else "w"))
fig.set_xticks(np.arange(len(xLabels)))
fig.set_yticks(np.arange(len(yLabels)))
fig.set_xticklabels(xLabels)
fig.set_yticklabels(yLabels)
plt.savefig("aciertoValNeuronal.pdf", format='pdf')
plt.show()

plt.figure(figsize=(10,10))
plt.title(r"Porcentaje de aciertos según el valor de $\lambda$ y " + 
          "el número de iteraciones")
plt.ylabel(r'$\lambda$')
plt.xlabel('Iteraciones')
fig = plt.subplot()
im = fig.imshow(acTrain, cmap=cm.viridis)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(xLabels)):
    for j in range(len(yLabels)):
        text = fig.text(i, j, round(acTrain[j][i],2), ha="center", 
               va="center", color=("k" if acTrain[j][i] > 93 else "w"))
fig.set_xticks(np.arange(len(xLabels)))
fig.set_yticks(np.arange(len(yLabels)))
fig.set_xticklabels(xLabels)
fig.set_yticklabels(yLabels)
plt.savefig("aciertoTrainNeuronal.pdf", format='pdf')
plt.show()
    
plt.figure(figsize=(10,10))
plt.title(r"Error según el valor de $\lambda$ y el número de iteraciones")
plt.ylabel(r'$\lambda$')
plt.xlabel('Iteraciones')
fig = plt.subplot()
im = fig.imshow(np.log10(errorVal), cmap=cm.viridis_r)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(xLabels)):
    for j in range(len(yLabels)):
        text = fig.text(i, j, round(np.log10(errorVal[j][i]),3), 
               ha="center", va="center", 
               color=("k" if np.log10(errorVal[j][i]) < -5 else "w"))
fig.set_xticks(np.arange(len(xLabels)))
fig.set_yticks(np.arange(len(yLabels)))
fig.set_xticklabels(xLabels)
fig.set_yticklabels(yLabels)
plt.savefig("errorValNeuronal.pdf", format='pdf')
plt.show()    
    
plt.figure(figsize=(10,10))
plt.title(r"Error según el valor de $\lambda$ y el número de iteraciones")
plt.ylabel(r'$\lambda$')
plt.xlabel('Iteraciones')
fig = plt.subplot()
im = fig.imshow(np.log10(errorTrain), cmap=cm.viridis_r)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(xLabels)):
    for j in range(len(yLabels)):
        text = fig.text(i, j, round(np.log10(errorTrain[j][i]),3), 
               ha="center", va="center", 
               color=("k" if np.log10(errorTrain[j][i]) < -5 else "w"))
fig.set_xticks(np.arange(len(xLabels)))
fig.set_yticks(np.arange(len(yLabels)))
fig.set_xticklabels(xLabels)
fig.set_yticklabels(yLabels)
plt.savefig("errorTrainNeuronal.pdf", format='pdf')
plt.show()  

# Calculamos el porcentaje de acierto sobre los datos de test para el
# valor escogido del término de regularización y de iteraciones
theta = minimize(fun=P2backprop, x0=theta0, 
        args=((np.shape(Xtrain)[1],size2,1), Xtrain, Ytrain2, 
        optReg), method='TNC', jac=True, options={'maxiter':optItera})['x']

theta1 = np.reshape(theta[:size2*(np.shape(Xtrain)[1]+1)],
                                 (size2,np.shape(Xtrain)[1]+1))
theta2 = np.reshape(theta[size2*(np.shape(Xtrain)[1]+1):],(1,size2+1))
        
resTest =  P2applyNet(Xtest, (theta1, theta2))[0]
ac = P2porc_ac(resTest, Ytest2)
print('El porcentaje de acierto sobre los datos de test es', ac, '%')


#%% Parte 3: Máquinas de soporte vectorial
    

# Comenzamos usando kernel lineal y distintos valores de C
parValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
acTrain = []
acVal = []
for C in parValues:
    svm = SVC(kernel='linear', C=C)
    svm.fit(Xtrain,Ytrain)
    
    acTrain.append(svm.score(Xtrain,Ytrain) * 100)
    acVal.append(svm.score(Xval,Yval) * 100)

# Comprobamos el porcentaje de acierto según el valor de C
opt = np.argmax(acVal)
print('El valor óptimo de C es', parValues[opt])

plt.figure(figsize=(10,10))
plt.plot(range(len(parValues)), acTrain, 'r', label="Entrenamiento")
plt.plot(range(len(parValues)), acVal, 'b', label="Validación")
plt.title(r"Porcentaje de acierto según $C$")
plt.xlabel(r"Valor de $C$")
plt.xticks(range(len(parValues)), parValues)
plt.ylabel("Porcentaje de acierto")
plt.legend(loc="lower left")
plt.savefig("aciertoSVM.pdf", format='pdf')
plt.show()
    
# Calculamos el porcentaje de acierto sobre los datos de test para el
# valor escogido de C
svm = SVC(kernel='linear', C=parValues[opt])
svm.fit(Xtrain,Ytrain)
ac = svm.score(Xtest,Ytest) * 100
print('El porcentaje de acierto sobre los datos de test es', ac, '%')
    
# Probamos ahora con kernel gaussiano utilizando distintos valores de C
# y de sigma
acTrain = np.zeros((len(regValues), len(regValues)))
acVal = np.zeros((len(regValues), len(regValues)))
for i in range(len(parValues)):
    for j in range(len(parValues)):
        svm = SVC(kernel='rbf', C=parValues[i], 
                  gamma=1/(2*parValues[j]**2))    
        svm.fit(Xtrain,Ytrain)
    
        acTrain[i][j] = svm.score(Xtrain,Ytrain) * 100
        acVal[i][j] = svm.score(Xval,Yval) * 100
    
# Comprobamos el pocentaje de acierto según los valores de C y sigma
opt = np.argmax(acVal)
optC, optSigma = parValues[opt//len(parValues)], parValues[opt%len(parValues)]
print('El valor óptimo del parámetro C es', optC)
print('El valor óptimo para el parámetro sigma es', optSigma)    
    
plt.figure(figsize=(10,10))
plt.title(r"Porcentaje de aciertos según los valores de $\sigma$ y C")
plt.ylabel('$C$')
plt.xlabel(r'$\sigma$')
fig = plt.subplot()
im = fig.imshow(acTrain, cmap=cm.viridis)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(parValues)):
    for j in range(len(parValues)):
        text = fig.text(i, j, round(acTrain[j][i],2), ha="center", 
            va="center", color=("k" if acTrain[j][i] > 70 else "w"))
fig.set_xticks(np.arange(len(parValues)))
fig.set_yticks(np.arange(len(parValues)))
fig.set_xticklabels(parValues)
fig.set_yticklabels(parValues)
plt.savefig("aciertoTrainSVM.pdf", format='pdf')
plt.show()    

plt.figure(figsize=(10,10))
plt.title(r"Porcentaje de aciertos según los valores de $\sigma$ y C")
plt.ylabel('$C$')
plt.xlabel(r'$\sigma$')
fig = plt.subplot()
im = fig.imshow(acVal, cmap=cm.viridis)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(parValues)):
    for j in range(len(parValues)):
        text = fig.text(i, j, round(acVal[j][i],2), ha="center", 
            va="center", color=("k" if acVal[j][i] > 70 else "w"))
fig.set_xticks(np.arange(len(parValues)))
fig.set_yticks(np.arange(len(parValues)))
fig.set_xticklabels(parValues)
fig.set_yticklabels(parValues)
plt.savefig("aciertoValSVM.pdf", format='pdf')
plt.show()
    
# Calculamos el porcentaje de acierto sobre los datos de test para el
# valor escogido de C y sigma
svm = SVC(kernel='rbf', C=optC, gamma=1/(2*optSigma**2))
svm.fit(Xtrain,Ytrain)
ac = svm.score(Xtest,Ytest) * 100
print('El porcentaje de acierto sobre los datos de test es', ac, '%')
    
#%% Parte 4: K-Medias


# Entrenamos un K-Medias con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0).fit(Xtrain)
trainLabels = kmeans.labels_
valLabels = kmeans.predict(Xval)

# Comprobamos el pocentaje de acierto según la interpretación de las etiquetas
acTrainA = np.count_nonzero(trainLabels == Ytrain) / len(Ytrain) * 100
acTrainB = np.count_nonzero(trainLabels != Ytrain) / len(Ytrain) * 100
acValA = np.count_nonzero(valLabels == Yval) / len(Yval) * 100
acValB = np.count_nonzero(valLabels != Yval) / len(Yval) * 100
    
print('Interpretando las etiquetas de forma directa, el entrenamiento ' + 
      'obtiene un porcentaje de acierto del ', acTrainA, '%')
print('Interpretando las etiquetas de forma inversa, el entrenamiento ' + 
      'obtiene un porcentaje de acierto del ', acTrainB, '%')
print('Interpretando las etiquetas de forma directa, la validación ' + 
      'obtiene un porcentaje de acierto del ', acValA, '%')
print('Interpretando las etiquetas de forma inversa, la validación ' + 
      'obtiene un porcentaje de acierto del ', acValB, '%')

# Calculamos el porcentaje de acierto sobre los datos de test para la 
# interpretación escogida de las etiquetas
testLabels = kmeans.predict(Xtest)
ac = (np.count_nonzero(testLabels == Ytest) if acValA > acValB 
      else np.count_nonzero(testLabels != Ytest)) / len(Ytest) * 100

print('El porcentaje de acierto sobre los datos de test es', ac, '%')

#%% Parte 5: Reducción de dimensionalidad


# Aplicamos PCA y comprobamos la varianza explicada para cada componente
pca = PCA()
XtrainR = pca.fit_transform(Xtrain)
expVar = pca.explained_variance_ratio_
expVarAcum = [expVar[0]]
for i in range(1, len(expVar)):
    expVarAcum.append(expVar[i] + expVarAcum[-1])

print("La varianza explicada acumulada es:", np.array(expVarAcum) * 100)


#%% Alternativa: Eliminar la variable 'olor' correlacionada


print("A partir de este punto repetimos los experimentos eliminando la " + 
      "variable 'odor', ya que está fuertemente correlacionada con la " + 
      "variable objetivo")

# Preparamos un conjunto adicional de datos eliminando la variable 'odor'
odor_lab = []
for c in Xtrain.columns:
    if c[:4] == 'odor':
        odor_lab.append(c)
Wtrain = Xtrain.copy().drop(odor_lab, 1)
Wval = Xval.copy().drop(odor_lab, 1)
Wtest = Xtest.copy().drop(odor_lab, 1)
Wtrain2 = np.hstack((np.array([np.ones(len(Ytrain))]).T, Wtrain))
Wval2 = np.hstack((np.array([np.ones(len(Yval))]).T, Wval))
Wtest2 = np.hstack((np.array([np.ones(len(Ytest))]).T, Wtest))


#%% Parte 1b: Regresión logística


# Entrenamos la regresión con distintos valores para el término de 
# regularización
theta0 = np.zeros(np.shape(Wtrain2)[1])
regValues = range(-10, 4)
thetas = []
errorTrain = []
acTrain = []
errorVal = []
acVal = []
for reg in regValues:
    theta = fmin_tnc(func=P1coste, x0=theta0, fprime=P1gradiente, 
                     args=(Wtrain2, Ytrain, 10**reg))[0]
    thetas.append(theta)
    errorTrain.append(P1coste(theta, Wtrain2, Ytrain))
    acTrain.append(P1porc_ac(Wtrain2, Ytrain, theta))
    errorVal.append(P1coste(theta, Wval2, Yval))
    acVal.append(P1porc_ac(Wval2, Yval, theta))
    
# Comprobamos el error y el pocentaje de acierto según el término de
# regularización
opt = np.argmin(errorVal)
print('El valor óptimo del parámetro de regularización es', 
      10**regValues[opt])

plt.figure(figsize=(10,10))
plt.plot(regValues, acTrain, 'r', label="Entrenamiento")
plt.plot(regValues, acVal, 'b', label="Validación")
plt.title(r"Porcentaje de acierto según $\lambda$")
plt.xlabel(r"Valor de $\lambda = 10^x$")
plt.ylabel("Porcentaje de acierto")
plt.legend(loc="lower left")
plt.savefig("aciertoLogisticaNO.pdf", format='pdf')
plt.show()

plt.figure(figsize=(10,10))
plt.plot(regValues, errorTrain, 'r', label="Entrenamiento")
plt.plot(regValues, errorVal, 'b', label="Validación")
plt.title(r"Error según $\lambda$")
plt.xlabel(r"Valor de $\lambda = 10^x$")
plt.ylabel("Error")
plt.legend(loc="upper left")
plt.savefig("errorLogisticaNO.pdf", format='pdf')
plt.show()
    
# Calculamos el porcentaje de acierto sobre los datos de test para el
# valor escogido del término de regularización
ac = P1porc_ac(Wtest2, Ytest, thetas[opt])
print('El porcentaje de acierto sobre los datos de test es', ac, '%')


#%% Parte 2b: Redes neuronales


# Creamos unas matrices inciales con pesos aleatorios
size2 = 25
theta01 = P2randomWeights(np.shape(Wtrain)[1], size2)
theta02 = P2randomWeights(size2, 1)
theta0 = np.concatenate((theta01.ravel(), theta02.ravel()))

regValues = range(-6, 4)
itera = range(10, 110, 10)

errorTrain = np.zeros((len(regValues), len(itera)))
acTrain = np.zeros((len(regValues), len(itera)))
errorVal = np.zeros((len(regValues), len(itera)))
acVal = np.zeros((len(regValues), len(itera)))

for i in range(len(regValues)):
    for j in range(len(itera)):
        theta = minimize(fun=P2backprop, x0=theta0, 
                args=((np.shape(Wtrain)[1],size2,1), Wtrain, Ytrain2, 
                10**regValues[i]), method='TNC', jac=True, 
                options={'maxiter':itera[j]})['x']

        theta1 = np.reshape(theta[:size2*(np.shape(Wtrain)[1]+1)],
                            (size2,np.shape(Wtrain)[1]+1))
        theta2 = np.reshape(theta[size2*(np.shape(Wtrain)[1]+1):],
                            (1,size2+1))
        
        resTrain =  P2applyNet(Wtrain, (theta1, theta2))[0]
        acTrain[i][j] = P2porc_ac(resTrain, Ytrain2)
        resVal = P2applyNet(Wval, (theta1, theta2))[0]
        acVal[i][j] = P2porc_ac(resVal, Yval2)
        
        errorTrain[i][j] = P2coste(Ytrain2, resTrain, [theta1,theta2], 0)
        errorVal[i][j] = P2coste(Yval2, resVal, [theta1,theta2], 0)
        
# Comprobamos el error y el pocentaje de acierto según el término de
# regularización y el número de iteraciones
opt = np.argmin(errorVal)
optReg, optItera = 10**regValues[opt//len(itera)], itera[opt%len(itera)]
print('El valor óptimo del parámetro de regularización es', optReg)
print('El valor óptimo para el número de iteraciones es', optItera)

xLabels = [str(it) for it in itera]
yLabels = [r'$10^{' + str(r) + '}$' for r in regValues]

plt.figure(figsize=(10,10))
plt.title(r"Porcentaje de aciertos según el valor de $\lambda$ y el " + 
          "número de iteraciones")
plt.ylabel(r'$\lambda$')
plt.xlabel('Iteraciones')
fig = plt.subplot()
im = fig.imshow(acVal, cmap=cm.viridis)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(xLabels)):
    for j in range(len(yLabels)):
        text = fig.text(i, j, round(acVal[j][i],2), ha="center", 
               va="center", color=("k" if acVal[j][i] > 93 else "w"))
fig.set_xticks(np.arange(len(xLabels)))
fig.set_yticks(np.arange(len(yLabels)))
fig.set_xticklabels(xLabels)
fig.set_yticklabels(yLabels)
plt.savefig("aciertoValNeuronalNO.pdf", format='pdf')
plt.show()

plt.figure(figsize=(10,10))
plt.title(r"Porcentaje de aciertos según el valor de $\lambda$ y el " + 
          "número de iteraciones")
plt.ylabel(r'$\lambda$')
plt.xlabel('Iteraciones')
fig = plt.subplot()
im = fig.imshow(acTrain, cmap=cm.viridis)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(xLabels)):
    for j in range(len(yLabels)):
        text = fig.text(i, j, round(acTrain[j][i],2), ha="center", 
            va="center", color=("k" if acTrain[j][i] > 93 else "w"))
fig.set_xticks(np.arange(len(xLabels)))
fig.set_yticks(np.arange(len(yLabels)))
fig.set_xticklabels(xLabels)
fig.set_yticklabels(yLabels)
plt.savefig("aciertoTrainNeuronalNO.pdf", format='pdf')
plt.show()
    
plt.figure(figsize=(10,10))
plt.title(r"Error según el valor de $\lambda$ y el número de iteraciones")
plt.ylabel(r'$\lambda$')
plt.xlabel('Iteraciones')
fig = plt.subplot()
im = fig.imshow(np.log10(errorVal), cmap=cm.viridis_r)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(xLabels)):
    for j in range(len(yLabels)):
        text = fig.text(i, j, round(np.log10(errorVal[j][i]),3), 
            ha="center", va="center", 
            color=("k" if np.log10(errorVal[j][i]) < -5 else "w"))
fig.set_xticks(np.arange(len(xLabels)))
fig.set_yticks(np.arange(len(yLabels)))
fig.set_xticklabels(xLabels)
fig.set_yticklabels(yLabels)
plt.savefig("errorValNeuronalNO.pdf", format='pdf')
plt.show()    
    
plt.figure(figsize=(10,10))
plt.title(r"Error según el valor de $\lambda$ y el número de iteraciones")
plt.ylabel(r'$\lambda$')
plt.xlabel('Iteraciones')
fig = plt.subplot()
im = fig.imshow(np.log10(errorTrain), cmap=cm.viridis_r)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(xLabels)):
    for j in range(len(yLabels)):
        text = fig.text(i, j, round(np.log10(errorTrain[j][i]),3), 
            ha="center", va="center", 
            color=("k" if np.log10(errorTrain[j][i]) < -5 else "w"))
fig.set_xticks(np.arange(len(xLabels)))
fig.set_yticks(np.arange(len(yLabels)))
fig.set_xticklabels(xLabels)
fig.set_yticklabels(yLabels)
plt.savefig("errorTrainNeuronalNO.pdf", format='pdf')
plt.show()  

# Calculamos el porcentaje de acierto sobre los datos de test para el
# valor escogido del término de regularización y de iteraciones
theta = minimize(fun=P2backprop, x0=theta0, 
        args=((np.shape(Wtrain)[1],size2,1), Wtrain, Ytrain2, optReg),
        method='TNC', jac=True, options={'maxiter':optItera})['x']

theta1 = np.reshape(theta[:size2*(np.shape(Wtrain)[1]+1)],
                    (size2,np.shape(Wtrain)[1]+1))
theta2 = np.reshape(theta[size2*(np.shape(Wtrain)[1]+1):],(1,size2+1))
        
resTest =  P2applyNet(Wtest, (theta1, theta2))[0]
ac = P2porc_ac(resTest, Ytest2)
print('El porcentaje de acierto sobre los datos de test es', ac, '%')


#%% Parte 3b: Máquinas de soporte vectorial
    

# Comenzamos usando kernel lineal y distintos valores de C
parValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
acTrain = []
acVal = []
for C in parValues:
    svm = SVC(kernel='linear', C=C)
    svm.fit(Wtrain,Ytrain)
    
    acTrain.append(svm.score(Wtrain,Ytrain) * 100)
    acVal.append(svm.score(Wval,Yval) * 100)

# Comprobamos el porcentaje de acierto según el valor de C
opt = np.argmax(acVal)
print('El valor óptimo de C es', parValues[opt])

plt.figure(figsize=(10,10))
plt.plot(range(len(parValues)), acTrain, 'r', label="Entrenamiento")
plt.plot(range(len(parValues)), acVal, 'b', label="Validación")
plt.title(r"Porcentaje de acierto según $C$")
plt.xlabel(r"Valor de $C$")
plt.xticks(range(len(parValues)), parValues)
plt.ylabel("Porcentaje de acierto")
plt.legend(loc="lower left")
plt.savefig("aciertoSVMNO.pdf", format='pdf')
plt.show()
    
# Calculamos el porcentaje de acierto sobre los datos de test para el
# valor escogido de C
svm = SVC(kernel='linear', C=parValues[opt])
svm.fit(Wtrain,Ytrain)
ac = svm.score(Wtest,Ytest) * 100
print('El porcentaje de acierto sobre los datos de test es', ac, '%')
    
# Probamos ahora con kernel gaussiano utilizando distintos valores de C
# y de sigma
acTrain = np.zeros((len(regValues), len(regValues)))
acVal = np.zeros((len(regValues), len(regValues)))
for i in range(len(parValues)):
    for j in range(len(parValues)):
        svm = SVC(kernel='rbf', C=parValues[i], 
                  gamma=1/(2*parValues[j]**2))    
        svm.fit(Wtrain,Ytrain)
    
        acTrain[i][j] = svm.score(Wtrain,Ytrain) * 100
        acVal[i][j] = svm.score(Wval,Yval) * 100
    
# Comprobamos el pocentaje de acierto según los valores de C y sigma
opt = np.argmax(acVal)
optC, optSigma = parValues[opt//len(parValues)], parValues[opt%len(parValues)]
print('El valor óptimo del parámetro C es', optC)
print('El valor óptimo para el parámetro sigma es', optSigma)    
    
plt.figure(figsize=(10,10))
plt.title(r"Porcentaje de aciertos según los valores de $\sigma$ y C")
plt.ylabel('$C$')
plt.xlabel(r'$\sigma$')
fig = plt.subplot()
im = fig.imshow(acTrain, cmap=cm.viridis)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(parValues)):
    for j in range(len(parValues)):
        text = fig.text(i, j, round(acTrain[j][i],2), ha="center", 
            va="center", color=("k" if acTrain[j][i] > 70 else "w"))
fig.set_xticks(np.arange(len(parValues)))
fig.set_yticks(np.arange(len(parValues)))
fig.set_xticklabels(parValues)
fig.set_yticklabels(parValues)
plt.savefig("aciertoTrainSVMNO.pdf", format='pdf')
plt.show()    

plt.figure(figsize=(10,10))
plt.title(r"Porcentaje de aciertos según los valores de $\sigma$ y C")
plt.ylabel('$C$')
plt.xlabel(r'$\sigma$')
fig = plt.subplot()
im = fig.imshow(acVal, cmap=cm.viridis)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(parValues)):
    for j in range(len(parValues)):
        text = fig.text(i, j, round(acVal[j][i],2), ha="center", 
            va="center", color=("k" if acVal[j][i] > 70 else "w"))
fig.set_xticks(np.arange(len(parValues)))
fig.set_yticks(np.arange(len(parValues)))
fig.set_xticklabels(parValues)
fig.set_yticklabels(parValues)
plt.savefig("aciertoValSVMNO.pdf", format='pdf')
plt.show()
    
# Calculamos el porcentaje de acierto sobre los datos de test para el
# valor escogido de C y sigma
svm = SVC(kernel='rbf', C=optC, gamma=1/(2*optSigma**2))
svm.fit(Wtrain,Ytrain)
ac = svm.score(Wtest,Ytest) * 100
print('El porcentaje de acierto sobre los datos de test es', ac, '%')
    

#%% Parte 4b: K-Medias


# Entrenamos un K-Medias con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0).fit(Wtrain)
trainLabels = kmeans.labels_
valLabels = kmeans.predict(Wval)

# Comprobamos el pocentaje de acierto según la interpretación de las etiquetas
acTrainA = np.count_nonzero(trainLabels == Ytrain) / len(Ytrain) * 100
acTrainB = np.count_nonzero(trainLabels != Ytrain) / len(Ytrain) * 100
acValA = np.count_nonzero(valLabels == Yval) / len(Yval) * 100
acValB = np.count_nonzero(valLabels != Yval) / len(Yval) * 100
    
print('Interpretando las etiquetas de forma directa, el entrenamiento' + 
      ' obtiene un porcentaje de acierto del ', acTrainA, '%')
print('Interpretando las etiquetas de forma inversa, el entrenamiento' + 
      ' obtiene un porcentaje de acierto del ', acTrainB, '%')
print('Interpretando las etiquetas de forma directa, la validación ' + 
      'obtiene un porcentaje de acierto del ', acValA, '%')
print('Interpretando las etiquetas de forma inversa, la validación ' + 
      'obtiene un porcentaje de acierto del ', acValB, '%')

# Calculamos el porcentaje de acierto sobre los datos de test para la 
# interpretación escogida de las etiquetas
testLabels = kmeans.predict(Wtest)
ac = (np.count_nonzero(testLabels == Ytest) if acValA > acValB 
      else np.count_nonzero(testLabels != Ytest)) / len(Ytest) * 100

print('El porcentaje de acierto sobre los datos de test es', ac, '%')


#%% Parte 5b: Reducción de dimensionalidad


# Aplicamos PCA y comprobamos la varianza explicada para cada componente
pca = PCA()
WtrainR = pca.fit_transform(Wtrain)
expVar = pca.explained_variance_ratio_
expVarAcum = [expVar[0]]
for i in range(1, len(expVar)):
    expVarAcum.append(expVar[i] + expVarAcum[-1])

print("La varianza explicada acumulada es:", np.array(expVarAcum) * 100)