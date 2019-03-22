import numpy as np
import math
from scipy import stats
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, Imputer,LabelEncoder
from sklearn.model_selection import train_test_split
labelencoder_X = LabelEncoder()
import seaborn as sns

# from fancyimpute import KNN, SoftImpute
filename = '/Users/mikexie/Downloads/banking.csv'

dataset = pd.read_csv(filename)

mis1 = dataset[2:4].values
mis2 = dataset.iloc[[0],[0,3]]

dataset[2:4] = np.nan
dataset.iloc[[0],[0,3]] = np.nan
dataset = dataset.head(5000)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


y = dataset['y']
# print(y.head(4))
X = dataset.loc[:, dataset.columns != 'y'] ## 除了y之外的列

print(dataset.columns)
for i in range(len(dataset.columns)):
    if dataset.columns[i] == 'y':
        print(i)



ind = np.where(y.isnull() == True)[0] # y 所在的缺失值的位置
X = X.drop(ind)
y = y.drop(ind)
# np.where(y[y.isnull() == True])
# print(ind, X.head(5))

for i in range(0, X.shape[1]): ##
    X.iloc[:,i].fillna(X.iloc[:,i].mode()[0], inplace=True) ## 众数填充

# nan_all = dataset.isnull().any()
# df2.drop(df2.columns[0], axis=1) ##

X = pd.get_dummies(X)

# print(X.describe)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) ###

# Method 1: Logistic regression
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = LogisticRegression()
# Train the model using the training sets and check score
model.fit(X_train, y_train)
model.score(X_train, y_train)

# Equation coefficient and Intercept
logit_model= sm.Logit(y_train,X_train)
result=logit_model.fit()

# print(result.summary())
# print('Coefficient: n', model.coef_)
# print('Intercept: n', model.intercept_)

#Predict Output
predicted = model.predict(X_test)
count = np.count_nonzero(predicted - y_test)
print(count)
print(count / len(predicted)) # error rate


# Method 2: KNN
model1 = KNeighborsClassifier(n_neighbors=3)
model1.fit(X_train, y_train)
score1 = model1.score(X_test, y_test)

predicted = model1.predict(X_test)
count = np.count_nonzero(predicted - y_test)
print(count)
print(count / len(predicted)) # error rate


