"""
ADS-1 Assignment 3: Clustering and Fitting

Author: Md Nadimozzaman Pappo <mnpappo@gmail.com>
Used Python Version: 3.11.4
Format: Black
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn import cluster
from cluster_tools import (
    scaler,
    backscale,
    logistic,
    cluster_number,
    map_corr,
)
from errors import err_ranges
from data_processing import reading_data
import importlib


def get_clusters_and_centers(df, ncluster, y1, y2, year=1990):
    """clusters_and_centers will plot clusters and its centers for given data

    Args:
        df (pd.DF): dataframe for which clustering is performed
        ncluster (int): number of clusters
        y1 (pd.DF): column name for x axis
        y2 (pd.DF): column name for y axis
        y (pd.DF): column name for year

    Returns:
        df: returns dataframe with labels
        centres: returns cluster centers
    """
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df)

    labels = kmeans.labels_
    df["labels"] = labels
    # extract the estimated cluster centres
    centres = kmeans.cluster_centers_

    centres = np.array(centres)
    xcen = centres[:, 0]
    ycen = centres[:, 1]

    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))

    sc = plt.scatter(df[y1], df[y2], 10, labels, marker="o")
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel(f"Electricity production from nuclear sources")
    plt.ylabel(f"Electricity production from nuclear sources")
    plt.legend(*sc.legend_elements(), title="clusters")
    plt.title(f"Clusters of Electricity production from nuclear sources in {year}")
    plt.show()

    return df, centres


def forecast_energy(data, country, start_year, end_year):
    """


    Parameters
    ----------
    data : pandas.DataFrame
        Data for which forecasting analysis is performed.
    country : STR
        Country for which forecasting is performed.
    start_year : INT
        Starting year for forecasting.
    end_year : INT
        Ending year for forecasting.

    Returns
    -------
    None.

    """
    data = data.loc[:, country]
    data = data.dropna(axis=0)

    energy = pd.DataFrame()

    energy["Year"] = pd.DataFrame(data.index)
    energy["Energy"] = pd.DataFrame(data.values)
    energy["Year"] = pd.to_numeric(energy["Year"])
    importlib.reload(optimize)

    param, covar = optimize.curve_fit(
        logistic, energy["Year"], energy["Energy"], p0=(1.2e12, 0.03, 1990.0)
    )

    sigma = np.sqrt(np.diag(covar))

    year = np.arange(start_year, end_year)
    forecast = logistic(year, *param)
    low, up = err_ranges(year, logistic, param, sigma)
    plt.figure()
    plt.plot(energy["Year"], energy["Energy"], label="Nuclear Energy Production")
    plt.plot(year, forecast, label="Forecast", color="k")
    plt.fill_between(year, low, up, color="green", alpha=0.3, label="Confidence Margin")
    plt.xlabel("Year")
    plt.ylabel("Electricity production from nuclear sources")
    plt.legend()
    plt.title(f"Electricity production from nuclear sources forecast for {country}")
    plt.savefig(f"./output/{country}.png", bbox_inches="tight", dpi=300)
    plt.show()

    energy2030 = logistic(2030, *param) / 1e9

    low, up = err_ranges(2030, logistic, param, sigma)
    sig = np.abs(up - low) / (2.0 * 1e9)
    print(
        f"Nuclear Energy Production by 2030 in {country}",
        np.round(energy2030 * 1e9, 2),
        "+/-",
        np.round(sig * 1e9, 2),
    )


# Reading data from csv file
energy, energy_t = reading_data("API_EG.ELC.NUCL.ZS_DS2_en_csv_v2_6304345.csv")

# Selecting years for which correlation is done for further analysis
energy = energy[["1990", "1995", "2000", "2005", "2010", "2015"]]
# print(energy.describe())

# map_corr(energy)
column1 = "1990"
column2 = "2015"

# Extracting columns for clustering
energy_ex = energy[[column1, column2]]
energy_ex = energy_ex.dropna(axis=0)

# Normalising data and storing minimum and maximum
energy_norm, df_min, df_max = scaler(energy_ex)


# print("Number of Clusters and Scores")
ncluster = cluster_number(energy_ex, energy_norm)

_, centers = get_clusters_and_centers(energy_norm, ncluster, column1, column2, 1990)
get_clusters_and_centers(energy_ex, ncluster, column1, column2, 2015)

# Applying backscaling to get actual cluster centers
actual_centers = backscale(centers, df_min, df_max)
# print("actual_centers: ", actual_centers)

print(energy_ex[energy_ex["labels"] == 0].index.values)
print(energy_ex[energy_ex["labels"] == 1].index.values)
print(energy_ex[energy_ex["labels"] == 2].index.values)
# Forecast Energy Production per capita for United Kingdom (Cluster 1),
# United States (Cluster 2) and Korea, Rep. (Cluster 3)
forecast_energy(energy_t, "Italy", 1970, 2030)
forecast_energy(energy_t, "United States", 1970, 2030)
forecast_energy(energy_t, "Korea, Rep.", 1970, 2030)
