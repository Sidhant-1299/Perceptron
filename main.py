
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


X = pd.concat([X1,X2]).to_numpy()
y = pd.concat([setosa['variety'],versicolor['variety']]).map({'Setosa':1,'Versicolor':0}).to_numpy()




"""
From the plot we can see a linear decision boundary sufficient to separate versicolor from setosa
Thus a linear classifier such as perceptron should be able to classify the flowers
"""

ppn = Perceptron(eta=0.1,n_iter=15)
ppn.fit(X,y)


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,7))
ax1.scatter(setosa['sepal.length'],setosa['petal.length'],color='r',label="Setosa")
ax1.scatter(versicolor['sepal.length'],versicolor['petal.length'],color='b',label="Versicolor")
ax1.set_xlabel("Sepal length (cm)")
ax1.set_ylabel("Petal length (cm)")
ax1.set_title("Setosa vs Versicolor ")
ax1.legend()

ax2.plot([i for i in range(1,len(ppn.error_) + 1)],ppn.error_)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Number of errors")
ax2.set_title("Percepton: Epochs vs Number of errors")


plt.tight_layout()
plt.show()

# ppn.export_weight()
# ppn.export_bias()