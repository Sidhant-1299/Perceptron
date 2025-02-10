
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron



iris_dataset_url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"

df = pd.read_csv(iris_dataset_url,encoding='utf-8')

setosa = df[df['variety']=="Setosa"]
versicolor = df[df['variety']=="Versicolor"]

X1 = setosa.loc[:,['sepal.length','petal.length']]
X2 = versicolor.loc[:,['sepal.length','petal.length']]


X = pd.concat([X1,X2])
y = pd.concat([setosa['variety'],versicolor['variety']])


fig,ax = plt.subplots()
ax.scatter(setosa['sepal.length'],setosa['petal.length'],color='r',label="Setosa")
ax.scatter(versicolor['sepal.length'],versicolor['petal.length'],color='b',label="Versicolor")
ax.set_xlabel("Sepal length (cm)")
ax.set_ylabel("Petal length (cm)")
plt.legend()
plt.show()

"""
From the plot we can see a linear decision boundary sufficient to separate versicolor from setosa
Thus a linear classifier such as perceptron should be able to classify the flowers
"""

# ppn = Perceptron(eta=0.1,n_iter=15)

