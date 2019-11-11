import numpy as np
import pandas as pd
import pandas_profiling
import prep
import seaborn as sns

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# df = prep.prep_df()
# df_2 = df.drop(columns = ["fips", "latitude", "longitude", "regionidcity", "regionidcounty", "regionidzip"])
# explore_df = pd.Series(df_2.corrwith(df["logerror"]))
# explore_df.nlargest(n=5)
# explore_df.nsmallest(n=5)
# #Seeing 5 largest and 5 smallest correlations. 
# profile = df.profile_report()
# rejected_variables = profile.get_rejected_variables(threshold=0.9)
# profile
# X_train, y_train, X_test, y_test = prep.get_train_test_split(df_2)
# from sklearn.linear_model import LinearRegression
# from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

# lm = LinearRegression()

# efs = EFS(lm, min_features=4, max_features=7, \
#     scoring='MSE', cv=10, n_jobs=-1)

# efs.fit(X_train, y_train)
# print('Best subset:', efs.best_idx_)
# df_2.columns
# Best Params = Bathrooms, Tax_value, lotsizesquarefeet


#Scaling and refitting. 
# from sklearn import preprocessing
# scaler = preprocessing.StandardScaler()
# scaled_log = scaler.fit_transform(df_2[["logerror"]])
# df_3 = df_2[["bathrooms", "tax_value", "lotsizesquarefeet", "logerror"]]
# train_2, test_2 = train_test_split(df_3, train_size = .75, random_state = 123)

# X_train_2 = train_2.drop(columns=["logerror"])
# y_train_2 = train_2["logerror"]
# X_test_2 = test_2.drop(columns=["logerror"])
# y_test_2 = test_2["logerror"]

# reg = lm.fit()






def scatter_plot(feature, target):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16,8))
    plt.scatter(df_3[feature], df_3[target], c="black")
    plt.xlabel(f"{feature}")
    plt.ylabel(f"{target}")
    plt.show()



    # scatter_plot("tax_value", "logerror")
    # scatter_plot("bathrooms", "logerror")
    # scatter_plot("lotsizesquarefeet", "logerror")
    # scatter_plot("yearbuilt", "logerror")

# funtion to cluster on y_train and merge back to train dataframe.
def target_cluster(y_train,X_train,num_clusters):
    kmeans =KMeans(n_clusters=num_clusters)
    kmeans.fit(y_train)
    y_train['cluster'] = kmeans.predict(y_train)
    train = X_train.merge(y_train,left_index=True,right_index=True)
    return train

def elbow_plot(target):
    ks = range(1,10)
    sse = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(target)

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_)

    print(pd.DataFrame(dict(k=ks, sse=sse)))

    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k')
    plt.show()

#function to cluster on X_train and merge back with train dataframe

def x_cluster(X_train,X_test,num_clusters):
    
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X_train)
    X_train['x_cluster'] = kmeans.predict(X_train)
    X_test['x_cluster'] = kmeans.predict(X_test)

    return X_train, X_test, kmeans


def bad_dist():
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
    def uneven_dist_chart_train():
        sns.distplot(y_train)
        plt.xlim(.4, .8)
        plt.show()
    def uneven_dist_chart_test():
        sns.distplot(y_test)
        plt.xlim(.4,.8)
        plt.show()
    return uneven_dist_chart_test, uneven_dist_chart_train
