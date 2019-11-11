import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import prep
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = prep.prep_df()
df["tax_per_sqft"] = df.tax_value/df.sqft
train, test = prep.get_train_and_test(df)
train.drop(columns=["tax_value", "sqft"], inplace=True)
test.drop(columns=["tax_value", "sqft"], inplace=True)
scaler = preprocessing.MinMaxScaler()
train.drop(columns=["latitude", "longitude"], inplace=True)
test.drop(columns=["latitude", "longitude"], inplace=True)
scaled_train = scaler.fit_transform(train[["logerror"]])
scaled_test = scaler.fit_transform(test[["logerror"]])
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

from sklearn.kernel_ridge import KernelRidge
reg = KernelRidge(alpha=1.0)
reg = reg.fit(X_train, y_train)
y_pred = reg.predict(X_train)
mean_squared_error(y_test, y_pred)


