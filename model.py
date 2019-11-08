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
        ("kmeans", KMeans(n_clusters=5, random_state = 123))])
    kmeans_pipeline.fit(train[cols])
    return silhouette_score(train[cols], kmeans_pipeline.predict(train[cols]))


score_sil(train, ["bedrooms"])
#score tax/log = 0.122
#score bedrooms/log = 0.738
#score bathrooms/log = 0.508
#score yearbuilt/log = 0.448
#score lotsizesquarefeet/log = -0.37
#score structuretaxvaluedollarcnt/log = 0.1212

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

t1 = train[["bathrooms", "logerror"]]
t1.groupby("bathrooms").mean().sort_values(by="logerror", ascending=True)

t2 = out[["price_per_sqft", "logerror"]]
t2.describe()
t1.groupby("sqft_avg").mean().sort_values(by="logerror", ascending=True)