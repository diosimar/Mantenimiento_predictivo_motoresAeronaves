# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 08:13:10 2022

@author: diosimarcardoza
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import SplineTransformer
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from utils.functions import timeEvolution


trainE = []
with open('Data/TrainSet.txt', 'r') as file:
    s = file.readlines()
    for i in s:
        trainE.append([np.float32(j) for j in i.split(" ")[: -2]])

testE = []
with open('Data/TestSet.txt', 'r') as file:
    s = file.readlines()
    for i in s:
        testE.append([np.float32(j) for j in i.split(" ")[: -2]])

result = []
with open('Data/TestSet_RUL.txt', 'r') as file:
    s = file.readlines()
    for i in s:
        result.append(int(i[: -2]))
result = np.array(result)


names = ["id", "cycle"] + ["setting_" + str(i) for i in range(3)] + ["s_" + str(i) for i in range(21)]
train = pd.DataFrame(trainE, columns=names)
test = pd.DataFrame(testE, columns=names)
#################################################################
## chack if the datasets have some null values 
train.info()
test.info()
#________________________________________________________________
n_train, n_features = train.shape
n_test = test.shape[0]
n_turb = result.shape[0]

train_failure = np.zeros(n_turb)
for i in range(n_turb):
    train_failure[i] = train[train['id'] == (i + 1)].cycle.max()
    
test_end = np.zeros(n_turb)
for i in range(n_turb):
    test_end[i] = test[test['id'] == (i + 1)].cycle.max()
    
#__ plotting__#
# hist of failded cycles for each engine in the training dataset
sns.displot(train_failure, color='#F2AB6D',  kde=True) #creamos el gráfico en Seaborn
plt.title('Histograma de tiempos maximos de  falla por turbina - train set')
plt.xlabel('Ciclo maximo en falla - Tiempo ')
plt.ylabel('Frecuencia')

# hist of executed cycles for each engine in the testing dataset
sns.displot(test_end, color='#F2AB6D',  kde=True) #creamos el gráfico en Seaborn
plt.title('Histograma de tiempos maximos de  falla por turbina - test set')
plt.xlabel('Ciclo maximo en falla - Tiempo ')
plt.ylabel('Frecuencia')
####################################################
#_________________
# smooth curve and scatterplot 

spline = SplineTransformer(degree=2, n_knots=3)

x_ = np.array(range(1, n_turb + 1))
X = spline.fit_transform(x_.reshape(-1,1))
# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()
# Entrenamos nuestro modelo
regr.fit(X, train_failure)
# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X)

plt.figure()
plt.plot(range(1, n_turb + 1), train_failure, 'bo')
plt.plot(x_,y_pred,'r--')
plt.xlabel("Numero de turbinas")
plt.ylabel("ciclo de  falla")
plt.title("ciclo de falla de las  turbinas")
plt.show()
#____________________________________________
#  boxplot to each sensor 
columns = train.columns.difference(['id','cycle','setting_0', 'setting_1', 'setting_2'])
df = train[columns]
descriptive_train = df.describe()

min_max_scaler = MinMaxScaler()
df = pd.DataFrame(min_max_scaler.fit_transform(df),
                  columns=columns, index=train.index)

join_data = train[train.columns.difference(columns)].join(df)

plt.boxplot(df)

#_________________________________________________
# check the trends over each sensor
def trend_plot(n , data):
    for i in range(1, n):
        k = 1
        plt.figure(1)
        for j in data.columns[5:]:
            plt.subplot(4, 6,k)
            plt.plot(join_data[join_data["id"] == i].cycle, join_data[join_data["id"] == i][j])
            plt.title('Sensor ' + str(j))
            plt.subplots_adjust(top=2, bottom=0.1, left=0.1, right=2, hspace=0.6, wspace=0.8)
            k += 1

n = n_turb + 1
trend_plot(n, train)
  
describe_train = pd.Series(train_failure).describe()
describe_test = pd.Series(test_end).describe()
plt.boxplot(train_failure)
plt.title('Boxplot  ciclos de falla de los motores - train dataset')
#####
plt.boxplot(test_end)
plt.title('Boxplot  ciclos de falla de los motores - test dataset')
#_____________________________________________________________________-
cols = ['s_1','s_2', 's_3', 's_6', 's_7', 's_8','s_10',
       's_11', 's_12', 's_13', 's_14', 's_16', 's_19', 's_20']

corr = train[cols].corr()
corr.style.background_gradient(cmap='coolwarm')

plt.figure(figsize = [12,5])
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,linewidths=0
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);




from factor_analyzer import FactorAnalyzer

def FA(data):
    #data = wine_data.loc[:,'fixed_acidity':'alcohol']
    fa = FactorAnalyzer()
    n_factors = 6

    fa.set_params(n_factors= n_factors, rotation="varimax")
    fa.fit(data)

    loadings = pd.DataFrame(fa.loadings_, columns = ['Factor {}'.format(i) for i in range(1, n_factors + 1)])
    loadings.index = data.columns

    varianza = pd.DataFrame(fa.get_factor_variance(), columns = loadings.columns)
    varianza.index = ['SS Loadings', 'Proportional Var', 'Cumulative Var']
    
    return (loadings, varianza)

df = join_data.loc[:, cols]
(loadings, varianza) = FA(df)
loadings
varianza
