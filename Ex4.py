from datetime import date

import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import  neighbors,datasets
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest,chi2
import time

#load dữ liệu lên trang
dataset= pd.read_csv("weatherAUS.csv")

#tiền xử lý dữ liệu
#xóa các cột các giá trị NA vì các cột này không góp phần training dữ liệu
dataset.drop(['Date','Location','Evaporation','Sunshine','Cloud9am','Cloud3pm'], axis='columns', inplace=True)
# nếu ô có chứa NA thì tiến hành xóa hàng đó
dataset = dataset.dropna(how='any')


#3. Xóa các ngoại lệ của dữ liệu, sử dụng Z-core để nhân diện, phát hiện và xóa bỏ các ngoại lệ
z=np.abs(stats.zscore(dataset._get_numeric_data()))
dataset=dataset[(z<3).all(axis=1)]


##tiến hành thay thế các dữ liệu trừu tượng thành các chữ số toán học
dataset['RainToday'].replace({'No':0,'Yes':1},inplace=True)
dataset['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)

categorical_columns = ['WindGustDir','WindDir3pm','WindDir9am']

# chuyển đổi các cột giwof thành các cột hướng gió
dataset = pd.get_dummies(dataset, columns=categorical_columns)
dataset.iloc[4:9]

#6. Chuan hóa dữ liệu, sử dụng MinMaxScaler
scaler =preprocessing.MinMaxScaler()
scaler.fit(dataset)
dataset = pd.DataFrame(scaler.transform(dataset),index=dataset.index,columns=dataset.columns)
dataset.iloc[4:10]


# Tiến hành tìm ra các cột quan trọng nhất trong dataset
X= dataset.loc[:,dataset.columns != 'RainTomorrow'] #bỏ đi cột rainTomorrow
y=dataset[['RainTomorrow']]
selector = SelectKBest(chi2,k=3) # chọn ra 3 cột có ảnh hưởng nhiều nhất đến kết quả
selector.fit(X,y)
X_new=selector.transform(X)
print(X.columns[selector.get_support(indices=True)])

dataset=dataset[['Rainfall', 'Humidity3pm', 'RainToday']]
X=dataset[['Rainfall', 'Humidity3pm', 'RainToday']]

X_train,X_test,y_train,y_test= train_test_split( X,y,test_size=0.25)

clf = neighbors.KNeighborsClassifier(n_neighbors=1,p=2)
clf.fit(X_train,y_train.values.ravel())
y_pred=clf.predict(X_test)
score =accuracy_score(y_test,y_pred)

print("print result for 15 test data point ")
print ("predict labels: ", y_pred)
print("Ground truth: ", y_test )
print ("Score: ",score)

