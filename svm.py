# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 00:23:48 2019

@author: hanne
"""
import numpy as np 
import pandas as pd 
import datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def acuracia(clf,X,y):
   resultados = cross_val_predict(clf, X, y, cv=2)
   print('Resultados')
   print(resultados)
   #print (classification_report(classes,resultados,target_names=['0','1']))
   return accuracy_score(y,resultados)

data = pd.read_csv('ohlc1dia.csv', header=0,index_col='Datetime')
data['Volume_(BTC)'].fillna(value=0, inplace=True)
data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)
data['CloseOpen']=(data['Close']-data['Open'])
#print(data)
#1 é alta
#2 é empate
#0 é baixo
data['Status'] = np.where(data['CloseOpen'] >0, 1, np.where(data['CloseOpen'] <0,0,2))
#print(data['Status'].head)
print('Daddos do treinamento')
treino = np.array(data.iloc[:, 0:4])
print(treino) 	# features

classes = data['Status']
#treino com 80% e 20% para o teste
treino[:-700]
classes[:-700]
svm.SVC(kernel='rbf',C=100,gamma=0.01) #pode deixar o default que dará no mesmo
clf = svm.SVC().fit(treino[:-700],classes[:-700])

pip_3 = Pipeline([('scaler',StandardScaler()),('clf', svm.SVC(kernel='rbf'))])

treino[-700:]
pred=clf.predict(treino[-700:])
classes[-700:]

#matplotlib notebook
from matplotlib import style
style.use('ggplot')
SAMPLE_SIZE = 200
#sepal length vs sepal width
plt.xlabel('Open')
plt.ylabel('Close')
plt.legend(loc='best')
plt.title('Test - Close x Open - 1 Day')
#x=list(range(0,700,1))
#plt.plot(x,classes[-700:])

#area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
plt.scatter(treino[-700:,0], treino[-700:,3], c=clf.predict(treino[-700:]))
#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
#plt.scatter(treino[-700:,0],treino[-700:,1],)
plt.show
print(accuracy_score(classes[-700:], pred)) #acuracia 
print(accuracy_score(classes[-700:], pred, normalize=False))
print(classification_report(classes[-700:], pred, labels=[0,1, 2]))

#result=acuracia(clf,treino,classes)
#lista_C = [0.001, 0.01, 0.1, 1, 10,100]
#lista_gamma = [0.001, 0.01, 0.1, 1, 10, 100]
#parametros_grid = dict(clf__C=lista_C, clf__gamma=lista_gamma)
#grid = GridSearchCV(pip_3,parametros_grid, cv=5,scoring='accuracy')

#grid.fit(clf,classes)

#print(grid.grid_scores_)
