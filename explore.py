import numpy as np
import pandas as pd
import pandas_profiling
import prep
import seaborn as sns
import pandas_profiling
df = prep.prep_df()
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
df_2 = df.drop(columns = ["assessmentyear", "propertycountylandusecode", "fips", "latitude", "longitude", "regionidcity", "regionidcounty", "regionidzip"])
df_2 = df_2.dropna()
train, test = train_test_split(df_2, train_size = .75, random_state = 123)
X_train = train.drop(columns=["logerror"])
y_train = train["logerror"]
X_test = test.drop(columns=["logerror"])
y_test = test["logerror"]
##SPLIT LAT AND LONG FROM TRAIN AND TEST AFTER SPLIT. 
df_2.astype("int")
df
##
lm = LinearRegression()

efs = EFS(lm, min_features=3, max_features=7, \
    scoring='neg_mean_squared_error', cv=10)

efs.fit(X_train, y_train)

print('Best subset:', efs.best_idx_)

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

from sklearn.ensemble import RandomForestRegressor
knn = RandomForestRegressor()
efs1 = EFS(knn, 
           min_features=1,
           max_features=4,
           scoring='accuracy',
           cv=5)
efs1 = efs1.fit(X_train, y_train)









