import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest,chi2
import time

#load file data can train vao trong chuong trinh
df =pd.read_csv('weatherAUS.csv')
#Tiến hành tiền xử lý dữ liệu
#Tiến hành xóa các cột có dự liệu đa số là NA vì những cột này không góp phần trong việc train và test dữ liệu
df.drop(['Date','Location','Evaporation','Sunshine','Cloud9am','Cloud3pm'], axis='columns', inplace=True)

#nếu ô nòa cố dữ liệu có chữ NA thì tiến hành xóa hàng đó
df = df.dropna(how='any')


#
#3. Xóa các ngoại lệ của dữ liệu, sử dụng Z-core để nhân diện, phát hiện và xóa bỏ các ngoại lệ
z=np.abs(stats.zscore(df._get_numeric_data()))
df=df[(z<3).all(axis=1)]
##tiến hành thay thế các dữ liệu trừu tượng thành các chữ số toán học
df['RainToday'].replace({'No':0,'Yes':1},inplace=True)
df['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)

categorical_columns = ['WindGustDir','WindDir3pm','WindDir9am']

#chuyển đổi các cột giờ thành các cột trong hướng gió
df = pd.get_dummies(df, columns=categorical_columns)
df.iloc[4:9]


#6. Chuan hóa dữ liệu, sử dụng MinMaxScaler
scaler =preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df),index=df.index,columns=df.columns)
df.iloc[4:10]

# Tiến hành tìm ra các cột quan trọng nhất trong dataset
X= df.loc[:,df.columns != 'RainTomorrow'] #bỏ đi cột rainTomorrow
y=df[['RainTomorrow']]
selector = SelectKBest(chi2,k=3) # chọn ra 3 cột có ảnh hưởng nhiều nhất đến kết quả
selector.fit(X,y)
X_new=selector.transform(X)
print(X.columns[selector.get_support(indices=True)])


df=df[['Rainfall', 'Humidity3pm', 'RainToday']]
X=df[['Rainfall', 'Humidity3pm', 'RainToday']]
# y=df[['RainTomorrow']]

X_train,X_test,y_train,y_test= train_test_split( X,y,test_size=0.25)

clf_dt= DecisionTreeClassifier(random_state=0)
t0=time.time()

clf_dt.fit(X_train,y_train)

y_pred =clf_dt.predict(X_test)
score =accuracy_score(y_test,y_pred)

print('Acuracy : ',score)
print(y_pred)
print('Time taken : ', time.time()-t0)