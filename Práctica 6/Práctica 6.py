'''
Práctica 6

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import codecs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.svm import SVC
from scipy.io import loadmat

os.chdir("./resources")

from process_email import email2TokenList
from get_vocab_dict import getVocabDict


#%% Parte 1.1

# Leemos los datos
data = loadmat('ex6data1.mat')
y = data['y'].ravel()
X = data['X']

# Entrenamos la SVM con kernel lineal y C=1 y 
# representamos el resultado
svm = SVC(kernel='linear', C=1)
svm.fit(X,y)

meshX1 = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 1000)
meshX2 = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 1000)
meshX1, meshX2 = np.meshgrid(meshX1, meshX2)
meshY = svm.predict(np.array([meshX1.ravel(), meshX2.ravel()]).T).reshape(meshX1.shape)

plt.figure(figsize=(10,10))
plt.scatter(X[y==0,0], X[y==0,1], c='r', marker='o')
plt.scatter(X[y==1,0], X[y==1,1], c='b', marker='o')
plt.contour(meshX1, meshX2, meshY)
plt.title("SVM con $C=1$")
plt.savefig('P1.1C1.png')
plt.show()

# Entrenamos la SVM con C=100 y representamos el resultado
svm = SVC(kernel='linear', C=100)
svm.fit(X,y)

meshY = svm.predict(np.array([meshX1.ravel(), meshX2.ravel()]).T).reshape(meshX1.shape)

plt.figure(figsize=(10,10))
plt.scatter(X[y==0,0], X[y==0,1], c='r', marker='o')
plt.scatter(X[y==1,0], X[y==1,1], c='b', marker='o')
plt.contour(meshX1, meshX2, meshY)
plt.title("SVM con $C=100$")
plt.savefig('P1.1C100.png')
plt.show()


#%% Parte 1.2

# Leemos los datos
data = loadmat('ex6data2.mat')
y = data['y'].ravel()
X = data['X']
         
# Entrenamos la SVM con kernel gaussiano, C=1 y sigma=0.1 y
# representamos el resutlado
svm = SVC(kernel='rbf', C=1, gamma=1/(2*0.1**2))
svm.fit(X,y)

meshX1 = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 1000)
meshX2 = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 1000)
meshX1, meshX2 = np.meshgrid(meshX1, meshX2)
meshY = svm.predict(np.array([meshX1.ravel(), meshX2.ravel()]).T).reshape(meshX1.shape)

plt.figure(figsize=(10,10))
plt.scatter(X[y==0,0], X[y==0,1], c='r', marker='o')
plt.scatter(X[y==1,0], X[y==1,1], c='b', marker='o')
plt.contour(meshX1, meshX2, meshY)
plt.title("SVM con $C=1$")
plt.savefig('P1.2.png')
plt.show()


#%% Parte 1.3

# Leemos los datos
data = loadmat('ex6data3.mat')
y = data['y'].ravel()
yval = data['yval'].ravel()
X = data['X']
Xval = data['Xval']

# Entrenamos la SVM con kernel gaussiano y distintos valores para
# C y sigma, almacenando el porcentaje de acierto para cada uno
values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
acc = np.empty((len(values),len(values)))
for i in range(len(values)):
    for j in range(len(values)):
        svm = SVC(kernel='rbf', C=values[i], gamma=1/(2*values[j]**2))
        svm.fit(X,y)
        acc[i][j] = svm.score(Xval, yval)
        
# Representamos la matriz de porcentaje de aciertos según el valor
# de C y sigma
xLabels = [str(v) for v in values]
yLabels = [str(v) for v in values]
plt.figure(figsize=(10,10))
plt.title(r"Porcentaje de aciertos según el valor de $C$ y $\sigma$")
plt.ylabel('$C$')
plt.xlabel(r'$\sigma$')
fig = plt.subplot()
im = fig.imshow(acc, cmap=cm.viridis)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(xLabels)):
    for j in range(len(yLabels)):
        text = fig.text(i, j, acc[j][i]*100, ha="center", va="center", color=("k" if acc[j][i]>0.7 else "w"))
fig.set_xticks(np.arange(len(xLabels)))
fig.set_yticks(np.arange(len(yLabels)))
fig.set_xticklabels(xLabels)
fig.set_yticklabels(yLabels)
plt.savefig("P1.3Cuadros.png")
plt.show()
        
# Representamos el resultado de la SVM con mayor precisión
m = np.argmax(acc)
C = values[m//len(values)]
sigma = values[m%len(values)]
svm = SVC(kernel='rbf', C=C, gamma=1/(2*sigma**2))
svm.fit(X,y)

meshX1 = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 1000)
meshX2 = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 1000)
meshX1, meshX2 = np.meshgrid(meshX1, meshX2)
meshY = svm.predict(np.array([meshX1.ravel(), meshX2.ravel()]).T).reshape(meshX1.shape)

plt.figure(figsize=(10,10))
plt.scatter(X[y==0,0], X[y==0,1], c='r', marker='o')
plt.scatter(X[y==1,0], X[y==1,1], c='b', marker='o')
plt.contour(meshX1, meshX2, meshY)
plt.title(r"SVM con $C="+str(C)+"$ y $\sigma="+str(sigma)+"$")
plt.savefig('P1.3.png')
plt.show()


#%% Parte 2

# Cargamos el diccionario de palabras
dic = getVocabDict()

# Leemos y procesamos los datos correspondientes a spam
spam = np.zeros((len(os.listdir('spam')), len(dic)))
for i, filename in enumerate(os.listdir('spam')):
    email_contents = codecs.open('spam/'+filename, 'r', encoding='utf-8', errors='ignore').read()
    email_tokens = email2TokenList(email_contents)
    for token in email_tokens:
        if token in dic.keys():
            spam[i,dic[token]-1] = 1

# Leemos y procesamos los datos correspondientes al easy ham
easy = np.zeros((len(os.listdir('easy_ham')), len(dic)))
for i, filename in enumerate(os.listdir('easy_ham')):
    email_contents = codecs.open('easy_ham/'+filename, 'r', encoding='utf-8', errors='ignore').read()
    email_tokens = email2TokenList(email_contents)
    for token in email_tokens:
        if token in dic.keys():
            easy[i,dic[token]-1] = 1
            
# Leemos y procesamos los datos correspondientes al hard ham
hard = np.zeros((len(os.listdir('hard_ham')), len(dic)))
for i, filename in enumerate(os.listdir('hard_ham')):
    email_contents = codecs.open('hard_ham/'+filename, 'r', encoding='utf-8', errors='ignore').read()
    email_tokens = email2TokenList(email_contents)
    for token in email_tokens:
        if token in dic.keys():
            hard[i,dic[token]-1] = 1
            
# Dividimos los conjuntos para entrenamiento, validación y test
spamIndx = np.arange(len(spam))
easyIndx = np.arange(len(easy))
hardIndx = np.arange(len(hard))
np.random.shuffle(spamIndx)
np.random.shuffle(easyIndx)
np.random.shuffle(hardIndx)

div1 = 0.6
div2 = 0.2

spamTrain = spam[spamIndx[:int(div1*len(spamIndx))]]
spamVal = spam[spamIndx[int(div1*len(spamIndx)):int((div1+div2)*len(spamIndx))]]
spamTest = spam[spamIndx[int((div1+div2)*len(spamIndx)):]]
easyTrain = easy[easyIndx[:int(div1*len(easyIndx))]]
easyVal = easy[easyIndx[int(div1*len(easyIndx)):int((div1+div2)*len(easyIndx))]]
easyTest = easy[easyIndx[int((div1+div2)*len(easyIndx)):]]
hardTrain = hard[hardIndx[:int(div1*len(hardIndx))]]
hardVal = hard[hardIndx[int(div1*len(hardIndx)):int((div1+div2)*len(hardIndx))]]
hardTest = hard[hardIndx[int((div1+div2)*len(hardIndx)):]]
X = np.vstack((spamTrain, easyTrain, hardTrain))
Xval = np.vstack((spamVal, easyVal, hardVal))
Xtest = np.vstack((spamTest, easyTest, hardTest))
y = np.concatenate((np.ones(len(spamTrain)), np.zeros(len(easyTrain)+len(hardTrain))))
yval = np.concatenate((np.ones(len(spamVal)), np.zeros(len(easyVal)+len(hardVal))))
ytest = np.concatenate((np.ones(len(spamTest)), np.zeros(len(easyTest)+len(hardTest))))

# Entrenamos una SVM con kernel gaussiano y distintos valores
# para C y sigma, y almacenamos el porcentaje de acierto 
# con los datos de entrenamiento y los de validación
values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
accTrain = np.empty((len(values),len(values)))
accVal = np.empty((len(values), len(values)))
for i in range(len(values)):
    for j in range(len(values)):
        svm = SVC(kernel='rbf', C=values[i], gamma=1/(2*values[j]**2))
        svm.fit(X,y)
        accTrain[i][j] = svm.score(X, y)
        accVal[i][j] = svm.score(Xval, yval)
        
# Representamos la matriz de porcentaje de aciertos para los datos
# de validación según el valor de C y sigma
xLabels = [str(v) for v in values]
yLabels = [str(v) for v in values]
plt.figure(figsize=(10,10))
plt.title(r"Porcentaje de aciertos según el valor de $C$ y $\sigma$")
plt.ylabel('$C$')
plt.xlabel(r'$\sigma$')
fig = plt.subplot()
im = fig.imshow(accVal, cmap=cm.viridis)
cax = make_axes_locatable(fig).append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
for i in range(len(xLabels)):
    for j in range(len(yLabels)):
        text = fig.text(i, j, round(accVal[j][i]*100, 2), ha="center", va="center", color=("k" if accVal[j][i]>0.9 else "w"))
fig.set_xticks(np.arange(len(xLabels)))
fig.set_yticks(np.arange(len(yLabels)))
fig.set_xticklabels(xLabels)
fig.set_yticklabels(yLabels)
plt.savefig("P2Cuadros.png")
plt.show()

# Buscamos el mayor porcentaje de aciertos para los datos
# de validación
m = np.argmax(accVal)
C = values[m//len(values)]
sigma = values[m%len(values)]
print("La mejor SVM se construye con C =", C, "y sigma =", sigma)
print("Consigue un " + str(accTrain.ravel()[m] * 100) + "% de aciertos para los datos de entrenamiento.")
print("Consigue un " + str(accVal.ravel()[m] * 100) + "%  de aciertos para los datos de validación.")

# Entrenamos nuevamente la SVM para estos valores
svm = SVC(kernel='rbf', C=C, gamma=1/(2*sigma**2))
svm.fit(X,y)

# Comprobamos el porcentaje de acierto para los datos de test
accTest = svm.score(Xtest, ytest)
print("Consigue un " + str(accTest * 100) + "% de aciertos para los datos de test.")

# Comprobamos qué porcentaje de acierto tiene para el easy ham
# y para el hard ham por separado
accEasy = svm.score(easyTest, np.zeros(len(easyTest)))
accHard = svm.score(hardTest, np.zeros(len(hardTest)))
accSpam = svm.score(spamTest, np.ones(len(spamTest)))
print("Consigue un " + str(accEasy * 100) + "% de aciertos para el easy ham.")
print("Consigue un " + str(accHard * 100) + "% de aciertos para el hard ham.")
print("Consigue un " + str(accSpam * 100) + "% de aciertos para el spam.")
