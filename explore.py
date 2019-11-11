import numpy as np
import pandas as pd
import pandas_profiling
import prep
import seaborn as sns
import pandas_profiling
df = prep.prep_df()
df_2 = df.drop(columns = ["fips", "latitude", "longitude", "regionidcity", "regionidcounty", "regionidzip"])
explore_df = pd.Series(df_2.corrwith(df["logerror"]))
explore_df.nlargest(n=5)
explore_df.nsmallest(n=5)
#Seeing 5 largest and 5 smallest correlations. 
profile = df.profile_report()
rejected_variables = profile.get_rejected_variables(threshold=0.9)
profile
X_train, y_train, X_test, y_test = prep.get_train_test_split(df_2)
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

lm = LinearRegression()

efs = EFS(lm, min_features=4, max_features=7, \
    scoring='MSE', cv=10, n_jobs=-1)

efs.fit(X_train, y_train)
print('Best subset:', efs.best_idx_)
df_2.columns
# Best Params = Bathrooms, Tax_value, lotsizesquarefeet








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




