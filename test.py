import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import prep
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
df = prep.prep_df()
df["tax_per_sqft"] = df.tax_value/df.sqft
df["logbin"] = df.logerror.round(decimals=1)
df = df[df.logbin != 2.8]
df = df[df.logbin != -3.7]
df = df[df.logbin != -4.7]
df = df[df.logbin != 5.3]
df.logbin.value_counts()


X = df.drop(columns=["logerror"])
y = df[["logerror"]]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=42, stratify=df.logbin)
train = X_train.join(y_train)
test = X_test.join(y_test)
train.drop(columns=["tax_value", "sqft", "logbin"], inplace=True)
test.drop(columns=["tax_value", "sqft", "logbin"], inplace=True)
scaler = preprocessing.MinMaxScaler()
train.drop(columns=["latitude", "longitude"], inplace=True)
test.drop(columns=["latitude", "longitude"], inplace=True)
scaled_train = scaler.fit_transform(train[["logerror"]])
scaled_test = scaler.transform(test[["logerror"]])
train["logerror"] = scaled_train
test["logerror"] = scaled_test
X_train, y_train, X_test, y_test = prep.get_train_test_split(train, test)
#MinMaxScale Logerror, drop lat and long, split to train and test data on logerror. 
lm = LinearRegression()
regr = lm.fit(X_train, y_train)
ypred_train = regr.predict(X_train)
ypred_test = regr.predict(X_test)
mean_squared_error(y_train, ypred_train)
#output MSE for Train model is: 0.02817864224808966
mean_squared_error(y_test, ypred_test)

X_train
y_train


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(1,1, figsize=(5,3))
axes.plot(test.logerror, y_test, "bo", label = "actuals", alpha=0.5)
axes.plot(test.logerror, ypred_test, "ro", label="predictions", alpha=0.5)
plt.xlabel("train.logerror")
plt.ylabel("Logerror")
plt.legend()
plt.suptitle("Linear Regression")
plt.show()

fig, axes = plt.subplots(1,1, figsize=(5,3))
axes.plot(train.logerror, y_train, "bo", label = "actuals", alpha=0.5)
axes.plot(train.logerror, ypred_train, "ro", label="predictions", alpha=0.5)
plt.xlabel("train.logerror")
plt.ylabel("Logerror")
plt.legend()
plt.suptitle("Linear Regression")
plt.show()

def uneven_dist_chart_train():
    sns.distplot(y_train)
    plt.xlim(.4, .8)
    plt.show()
    
d
def uneven_dist_chart_test():
    sns.distplot(y_test)
    plt.xlim(.4,.8)
    plt.show()

