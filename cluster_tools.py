""" Tools to support clustering: correlation heatmap, normaliser and scale 
(cluster centres) back to original scale, check for mismatching entries """

import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn import metrics


def map_corr(df, size=10):
    """Function creates heatmap of correlation matrix for each pair of
    columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)

    The function does not have a plt.show() at the end so that the user
    can savethe figure.
    """

    import matplotlib.pyplot as plt  # ensure pyplot imported

    corr = df.corr()
    plt.figure(figsize=(size, size))
    plt.matshow(corr, cmap="coolwarm")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Nuclear Energy Production Correlation - Years vs Countries")
    plt.colorbar()
    plt.show()

    # plot the scatter matrix
    pd.plotting.scatter_matrix(df, figsize=(12, 12), s=5, alpha=0.8)
    plt.show()


def cluster_number(df, df_normalised):
    """cluster_number calculates the best number of clusters based on silhouette
    score

    Args:
        df (_type_): _description_
        df_normalised (_type_): _description_

    Returns:
        _type_: _description_
    """
    clusters = []
    scores = []
    # loop over number of clusters
    for ncluster in range(2, 10):
        # Setting up clusters over number of clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)

        # Cluster fitting
        kmeans.fit(df_normalised)
        lab = kmeans.labels_

        # Silhoutte score over number of clusters
        print(ncluster, metrics.silhouette_score(df, lab))

        clusters.append(ncluster)
        scores.append(metrics.silhouette_score(df, lab))

    clusters = np.array(clusters)
    scores = np.array(scores)
    best_ncluster = clusters[scores == np.max(scores)]
    # print("best n clusters", best_ncluster[0])

    return best_ncluster[0]


def scaler(df):
    """Expects a dataframe and normalises all
    columnsto the 0-1 range. It also returns
    dataframes with minimum and maximum for
    transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df - df_min) / (df_max - df_min)

    return df, df_min, df_max


def backscale(arr, df_min, df_max):
    """Expects an array of normalised cluster centres and scales
    it back. Returns numpy array."""

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr


def get_diff_entries(df1, df2, column):
    """Compares the values of column in df1 and the column with the same
    name in df2. A list of mismatching entries is returned. The list will be
    empty if all entries match."""

    import pandas as pd  # to be sure

    # merge dataframes keeping all rows
    df_out = pd.merge(df1, df2, on=column, how="outer")
    print("total entries", len(df_out))
    # merge keeping only rows in common
    df_in = pd.merge(df1, df2, on=column, how="inner")
    print("entries in common", len(df_in))
    df_in["exists"] = "Y"

    # merge again
    df_merge = pd.merge(df_out, df_in, on=column, how="outer")

    # extract columns without "Y" in exists
    df_diff = df_merge[(df_merge["exists"] != "Y")]
    diff_list = df_diff[column].to_list()

    return diff_list


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0
    and growth rate g"""

    function = n0 / (1 + np.exp(-g * (t - t0)))

    return function
