import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data =pd.read_csv('data.csv',header=None)
X= data.values
print(data)


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
result=imp.transform(X)
print("ket qua la: ")
print(result)
