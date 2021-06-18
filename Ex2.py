import pandas as pd
import numpy as np
import matplotlib.pyplot as pit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


#importing the dataset
dataset =  pd.read_csv('dataset.csv')
x = dataset.iloc[:,[2,3]].values
y =dataset.iloc[:,4].values


# chia dữ liệu thành các tập train va test
X_train, X_test, Y_train,Y_test= train_test_split(x,y,test_size=0.25,random_state= 0)

# chuẩn hóa dữ liệu
sc= StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fitting  classifier to the Training set
#create your classifier here
classifier =KNeighborsClassifier()
classifier.fit(X_train,Y_train)

# predicting the Test set result
y_pred = classifier.predict(X_test)

#Making the Confution Matrix
cm=confusion_matrix(Y_test,y_pred)

print("[INFO] feature examples: \n",x[0:5])
print("[INFO] label examples: \n",y[0:5])
