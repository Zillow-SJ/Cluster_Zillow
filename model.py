import pandas as pd
from prep import prep_df, get_train_test_split, target_cluster, get_train_and_test
df = prep_df()
df.drop(columns = ["fips", "regionidcity", "regionidcounty", "regionidzip"], inplace=True)
X_train, y_train, X_test, y_test = get_train_test_split(df)
train, test = get_train_and_test(df)
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def map_scat(train, cluster_by, n_clusters):
    kmeans_pipeline = Pipeline([("scale", StandardScaler()),\
     ("kmeans", KMeans(n_clusters=n_clusters, random_state = 123))])
    kmeans_pipeline.fit(train[[cluster_by]])
    clusters = kmeans_pipeline.predict(train[[cluster_by]])
    clusters = pd.DataFrame(clusters)
    clusters.rename(columns={0: "clusters"}, inplace=True)
    clusters.clusters = "clusters: " + clusters.clusters.astype("str") 
    train = train.reset_index()
    train.drop(columns=["index"], inplace=True)
    train = pd.merge(train, clusters, left_index=True, right_index=True)
    fig, axes = plt.subplots(1,1, figsize=(7,7))
    ax = sns.scatterplot(\
        train.latitude,
        train.longitude,
        hue=kmeans_pipeline.predict(train[[cluster_by]]))

    return plt.show()



map_scat(train, "logerror", 5)


def cluster(train, cluster_by, n_clusters):
    kmeans_pipeline = Pipeline([("scale", StandardScaler()),\
     ("kmeans", KMeans(n_clusters=n_clusters, random_state = 123))])
    kmeans_pipeline.fit(train[[cluster_by]])
    clusters = kmeans_pipeline.predict(train[[cluster_by]])
    clusters = pd.DataFrame(clusters)
    clusters.rename(columns={0: "clusters"}, inplace=True)
    clusters.clusters = "clusters: " + clusters.clusters.astype("str") 
    train = train.reset_index()
    train.drop(columns=["index"], inplace=True)
    train = pd.merge(train, clusters, left_index=True, right_index=True)
    return train


def score_sil(train, cols):
    from sklearn.metrics import silhouette_score
    kmeans_pipeline = Pipeline([("scale", StandardScaler()),\
        ("kmeans", KMeans(n_clusters=4, random_state = 123))])
    kmeans_pipeline.fit(train[cols])
    return silhouette_score(train[cols], kmeans_pipeline.predict(train[cols]))


score_sil(train, [ "price_sqft", "lotsizesquarefeet", "structuretaxvaluedollarcnt"])
#score tax/log = 0.122
#score bedrooms/log = 0.738
#score bathrooms/log = 0.508
#score yearbuilt/log = 0.448
#score sqft/log = -0.37
#score structuretaxvaluedollarcnt/log = 0.1212
#score structuretax/price_sqft/lotsize = 0.145 / 0.754(wit4) !!!!!!!!!!!!!!!!!!!!
#score price_sqft/lotsize = 0.0156 / 0.140(wit3)
#score structuretax/lotsize = 0.755 / 0.848(wit3)
#score structruetax/price_sqft = 0.116 / 0.232(wit3)


#ttest of tax value returned negligible resulets. 







kmeans_pipeline = Pipeline([("scale", StandardScaler()),\
    ("kmeans", KMeans(n_clusters=5, random_state = 123))])
kmeans_pipeline.fit(train[["logerror"]])
clusters = kmeans_pipeline.predict(train[["logerror"]])
clusters = pd.DataFrame(clusters)
clusters.rename(columns={0: "clusters"}, inplace=True)
clusters.clusters = "clusters: " + clusters.clusters.astype("str") 
train = train.reset_index()
train.drop(columns=["index"], inplace=True)
out = pd.merge(train, clusters, left_index=True, right_index=True)
out.clusters.value_counts()
out["price_per_sqft"] = out.tax_value/out.sqft
out.groupby('clusters').mean().sort_values(by='logerror')



#val counts for clusters
 
#average for 1,2,3,4 bathrooms then corr_average for logerror - ttest

t1 = train[["bedrooms", "logerror"]]
t1.groupby("bedrooms").mean().sort_values(by="logerror", ascending=True)

t2 = out[["price_per_sqft", "logerror"]]
t2["sqft_avg"] = pd.cut(t2["price_per_sqft"], bins=5)
t3 = t2.groupby("sqft_avg").mean()
from scipy.stats import ttest_ind, norm
t3.reset_index(inplace=True)
t3
ttest_ind(t3.loc[0].logerror,t3.loc[1].logerror)
t3.loc[1].logerror
t3.loc[0].logerror
t2["log_bin"] = pd.cut(t2["logerror"], bins = 5)
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

logerrors_above
logerrors_normal
logerrors_below


### DROP BATHROOMS, BEDROOMS, SQFT, TAX_VALUE, YEAR_BUILT 
train["price_sqft"] = train.tax_value/train.sqft
train.drop(columns=["bathrooms", "bedrooms", "sqft", "tax_value", "yearbuilt"], inplace =True)
train.columns
train.corr()
test["price_sqft"] = test.tax_value/test.sqft
test.drop(columns=["bathrooms", "bedrooms", "sqft", "tax_value", "yearbuilt"], inplace =True)
test.columns
#Scaling and refitting. 
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
train.drop(columns =["latitude", "longitude"], inplace = True)
test.drop(columns =["latitude", "longitude"], inplace =True)
scaled_log = scaler.fit_transform(train[["logerror"]])
scaled_log_test = scaler.fit_transform(test[["logerror"]])
train["logerror"] = scaled_log
test["logerror"] = scaled_log_test
X_train = train.drop(columns=["logerror"])
y_train = train["logerror"]
X_test = test.drop(columns=["logerror"])
y_test = test["logerror"]

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
reg = lm.fit(X_train, y_train)
ypred = reg.predict(X_train)
ypred_final = reg.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, ypred)
mean_squared_error(y_test, ypred_final)
reg.score(X_test, y_test)
baseline = y_train.values
baseline = pd.DataFrame(baseline)
baseline["mean_logerror"] = baseline[0].mean()
baseline = baseline.mean_logerror.values
#output from model is 0.02817864224808966
#output from scaled test model is 0.006084700981586714
import seaborn as sns
seaout = pd.DataFrame({'actual': y_train,
              'lm1': ypred.ravel(),
              'lm_baseline': baseline.ravel()})\
.melt(id_vars=['actual'], var_name='model', value_name='prediction')\
.pipe((sns.relplot, 'data'), x='actual', y='prediction', hue='model', palette = "Blues_d")
# y_train = pd.DataFrame(y_train)
# y_unscaled = pd.DataFrame(scaler.inverse_transform(y_train), columns=y_train.columns.values).set_index([y_train.index.values])
# y_train

from sklearn.linear_model import OrthogonalMatchingPursuit
orth = OrthogonalMatchingPursuit().fit(X_train, y_train)
ypred_orth = orth.predict(X_train)
y_pred_or = orth.predict(X_test)
mean_squared_error(y_train, ypred_orth)
mean_squared_error(y_test, y_pred_or)

#output from model is 0.028178642248089667

from sklearn import linear_model
clf = linear_model.Ridge(alpha=.5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
mean_squared_error(y_train, y_pred)
#output from model is 0.02817864224808966

from sklearn.linear_model import ElasticNet
clf2 = ElasticNet(random_state=123)
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_train)
mean_squared_error(y_train, y_pred2)
#output from model is 0.028179318168790338

reg2 = linear_model.BayesianRidge()
reg2.fit(X_train, y_train)
y_pred3 = reg2.predict(X_train)
mean_squared_error(y_train, y_pred3)
#output from model is 0.02817864224809334

