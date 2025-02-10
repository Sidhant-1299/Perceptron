
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
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


def plot_decision_regions(X, y, classifier,xlabel,ylabel,resolution=0.02,):
    markers = ('o', 's')
    colors = ('red', 'blue')
    cmap = ListedColormap(colors[:2]) 

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx],
                    label=f'Class {cl}', edgecolor='black')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()



xlabel = 'Sepal length [cm]'
ylabel = 'Petal length [cm]'

plot_decision_regions(X, y, classifier=ppn,xlabel=xlabel,ylabel=ylabel)

# ppn.export_weight()
# ppn.export_bias()