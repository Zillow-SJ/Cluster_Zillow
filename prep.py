import warnings
warnings.filterwarnings("ignore")
import acquire
import pandas as pd
import numpy as np
def drop_columns(df):
    df_nulls_c = pd.DataFrame(df.apply(lambda x: len(x) - x.count(),axis=0))
    df_nulls_c['pct_rows_missing'] = df_nulls_c[0] / len(df)
    column_drops = df_nulls_c[df_nulls_c['pct_rows_missing'] >.2]
    column_drops['column_names'] = column_drops.index
    column_drops = list(column_drops.column_names)
    df_new = df.drop(column_drops,axis=1)
    df_new = df_new.drop(columns=['parcelid','id','transactiondate',\
        'bathroomcnt','bedroomcnt','calculatedbathnbr','finishedsquarefeet12',\
            'censustractandblock','fullbathcnt','propertylandusetypeid',\
                'rawcensustractandblock','roomcnt','calculatedfinishedsquarefeet',\
                    'landtaxvaluedollarcnt','taxamount','taxvaluedollarcnt',\
                        'assessmentyear','propertycountylandusecode',"fips", "regionidcity", "regionidcounty", "regionidzip"])
    return df_new

def drop_rows(df):
    df_nulls_r = pd.DataFrame(df.apply(lambda x: df.shape[1] -x.count(),axis=1))
    df_nulls_r['pct_rows_missing'] = df_nulls_r[0] / df.shape[1]
    row_drops = df_nulls_r[df_nulls_r['pct_rows_missing'] >.2]
    row_drops['column_names'] = row_drops.index
    row_drops = list(row_drops.column_names)
    df_new = df.drop(row_drops,axis=0)
    return df_new



def prep_df():
    #query for single family residence from zillow db
    df = acquire.wrangle_zillow()

    #imputing values for NaNs
    df['tax_value_per_foot'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet
    df.lotsizesquarefeet = df['lotsizesquarefeet'].fillna((df.landtaxvaluedollarcnt /df.tax_value_per_foot.mean()))
    df = df.drop(columns='tax_value_per_foot')
    df.structuretaxvaluedollarcnt = df['structuretaxvaluedollarcnt'].fillna(df.tax_value * (df.structuretaxvaluedollarcnt/df.tax_value).mean())

    #drops columns with more than 20% missing values and columns with high correlation
    df = drop_columns(df)

    #drop rows with NAs. Less than 1100 rows of 55,000 data set.
    df = df.dropna(how='any')
    return df




# def impute_values(df):
#     df['tax_value_per_foot'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet
#     df.lotsizesquarefeet = df['lotsizesquarefeet'].fillna((df.landtaxvaluedollarcnt /df.tax_value_per_foot.mean()))
#     df = df.drop(columns='tax_value_per_foot')
#     df.structuretaxvaluedollarcnt = df['structuretaxvaluedollarcnt'].fillna(df.tax_value * (df.structuretaxvaluedollarcnt/df.tax_value).mean())
#     return df


def remove_outliers_iqr(df, columns):
    for col in columns:
        q75, q25 = np.percentile(df[col], [75,25])
        ub = 3*stats.iqr(df[col]) + q75
        lb = q25 - 3*stats.iqr(df[col])
        print(len(df[df[col] <= ub]))
        df = df[df[col] <= ub]
        df = df[df[col] >= lb]
    return df

def target_cluster(X_train, y_train):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5)
    y_train = pd.DataFrame([y_train])
    y_train = y_train.T
    kmeans.fit(y_train)
    y_train['cluster_y'] = kmeans.predict(y_train)
    train = X_train.join(y_train)
    return train

def get_train_test_split(df):
    from sklearn.model_selection import train_test_split
    df = df.dropna()
    train, test = train_test_split(df, train_size = .70, random_state = 123)
    X_train = train.drop(columns=["logerror"])
    y_train = train["logerror"]
    X_test = test.drop(columns=["logerror"])
    y_test = test["logerror"]
    return X_train, y_train, X_test, y_test

def get_train_and_test(df):
    from sklearn.model_selection import train_test_split
    df = df.dropna()
    train, test = train_test_split(df, train_size = .70, random_state = 123)
    train['tax_per_sqft'] = train.tax_value/train.sqft
    test['tax_per_sqft'] = test.tax_value/test.sqft
    return train, test

def get_baseline_train_test_split(df):
    from sklearn.model_selection import train_test_split
    df = df.dropna()
    train, test = train_test_split(df, train_size = .70, random_state = 123)
    X_train = train.drop(columns=["logerror",'latitude','longitude'])
    y_train = train["logerror"]
    y_train = pd.DataFrame(y_train)
    X_test = test.drop(columns=["logerror",'latitude','longitude'])
    y_test = test["logerror"]
    y_test = pd.DataFrame(y_test)
    return X_train, y_train, X_test, y_test