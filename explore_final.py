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
        plt.suptitle("Train Logerror Distribution")
        plt.show()
    def uneven_dist_chart_test():
        sns.distplot(y_test)
        plt.xlim(.4,.8)
        plt.suptitle("Test Logerror Distribution")
        plt.show()

    x = uneven_dist_chart_train()
    y = uneven_dist_chart_test()
    return x, y


def logerror_outliers():
    df = prep.prep_df_initial()
    train, test = prep.get_train_and_test(df)
    logerror_outliers = train[(train.logerror < -1)]
    logerrors_below = logerror_outliers.mean()
    logerrors_below = pd.DataFrame(logerrors_below)
    logerrors_below = logerrors_below.T
    logerrors_below
    from statistics import stdev
    logerrors_normal = train[(train.logerror < 0.03) | (train.logerror > -0.02)]
    logerrors_normal = logerrors_normal.mean()
    logerrors_normal = pd.DataFrame(logerrors_normal)
    logerrors_normal = logerrors_normal.T
    logerrors_normal
    logerror_outliers_above = train[(train.logerror > 1)]
    logerrors_above = logerror_outliers_above.mean()
    logerrors_above = pd.DataFrame(logerrors_above)
    logerrors_above = logerrors_above.T
    logerrors_above["price_sqft"] = logerrors_above.tax_value/logerrors_above.sqft
    logerrors_below["price_sqft"] = logerrors_below.tax_value/logerrors_below.sqft
    logerrors_normal["price_sqft"] = logerrors_normal.tax_value/logerrors_normal.sqft
    df = logerrors_above.append(logerrors_normal)
    df = df.append(logerrors_below)
    df.drop(columns='tax_per_sqft',inplace=True)
    df.index = ['logerrors<1', 'logerrors~0', 'logerrors>1']
    return df

def strat():
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
    def dist_chart_train():
        sns.distplot(y_train)
        plt.suptitle("Train Logerror Distribution")
        plt.show()
    def dist_chart_test():
        sns.distplot(y_test)
        plt.suptitle("Test Logerror Distribution")
        plt.show()
    x = dist_chart_train()
    y = dist_chart_test()
    return x, y


def strat_test_train():
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
    return train, test
    


def final_model():
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
    x = mean_squared_error(y_train, ypred_train)
    #output MSE for Train model is: 0.02817864224808966
    y = mean_squared_error(y_test, ypred_test)
    print (f"MSE on Train:{x}, MSE on Test {y}")
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