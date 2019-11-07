import numpy as np
import pandas as pd
import pandas_profiling
import prep
import seaborn as sns
import pandas_profiling
df = prep.prep_df()
df_2 = df.drop(columns = ["propertycountylandusecode", "fips", "latitude", "longitude", "regionidcity", "regionidcounty", "regionidzip"])
explore_df = pd.Series(df_2.corrwith(df["logerror"]))
explore_df.nlargest(n=5)
explore_df.nsmallest(n=5)
#Seeing 5 largest and 5 smallest correlations. 
profile = df.profile_report()
rejected_variables = profile.get_rejected_variables(threshold=0.9)
profile

from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.model_selection import train_test_split
df_2 = df_2.dropna()
train, test = train_test_split(df_2, train_size = .75, random_state = 123q)
X_train = train.drop(columns=["logerror"])
y_train = train["logerror"]
X_test = test.drop(columns=["logerror"])
y_test = test["logerror"]

lm = LinearRegression()

efs = EFS(lm, min_features=3, max_features=7, \
    scoring='r2', cv=10, n_jobs=-1)

efs.fit(X_train, y_train)
print('Best subset:', efs.best_idx_)
df_2.columns
# Best Params = Bathrooms, Tax_value, lotsizesquarefeet


#Scaling and refitting. 
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaled_log = scaler.fit_transform(df_2[["logerror"]])
df_3 = df_2[["bathrooms", "tax_value", "lotsizesquarefeet", 'yearbuilt', "logerror"]]
train_2, test_2 = train_test_split(df_3, train_size = .75, random_state = 123)

X_train_2 = train_2.drop(columns=["logerror"])
y_train_2 = train_2["logerror"]
X_test_2 = test_2.drop(columns=["logerror"])
y_test_2 = test_2["logerror"]








def scatter_plot(feature, target):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16,8))
    plt.scatter(df_3[feature], df_3[target], c="black")
    plt.xlabel(f"{feature}")
    plt.ylabel(f"{target}")
    plt.show()



scatter_plot("tax_value", "logerror")
scatter_plot("bathrooms", "logerror")
scatter_plot("lotsizesquarefeet", "logerror")
scatter_plot("yearbuilt", "logerror")

