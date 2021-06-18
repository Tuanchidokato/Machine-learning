import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import  neighbors,datasets
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("dataset-weather.csv")
dataset = dataset.dropna(how='any')
dataset['RainToday'].replace({'No':0,'Yes':1},inplace=True)
dataset['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)




iris_X=dataset.loc[:,dataset.columns != 'RainTomorrow']
iris_Y=dataset[['RainTomorrow']]
X_train,X_test,Y_train,Y_test= train_test_split(iris_X,iris_Y,test_size=0.2)

print("Training size: %d" %len(Y_train))
# print(Y_train)
print("Test size: %d" %len(Y_test))
# print(Y_test)

clf = neighbors.KNeighborsClassifier(n_neighbors=1,p=2)
clf.fit(X_train,Y_train.values.ravel())
y_pred=clf.predict(X_test)

print("print result for 15 test data point ")
print ("predict labels: ", y_pred)
print("Ground truth: ", Y_test )
