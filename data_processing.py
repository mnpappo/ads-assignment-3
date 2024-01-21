import pandas as pd


def reading_data(filepath):
    """Reading data from csv file

    Args:
        filepath (sting): filepath of csv file

    Returns:
        pd.DataFrame: returns pandas dataframe
        pd.Dataframe: returns pandas dataframe transposed
    """
    df = pd.read_csv(filepath, skiprows=4)
    df = df.set_index("Country Name", drop=True)
    df = df.loc[:, "1960":"2021"]

    df_t = df.transpose()

    return df, df_t


def preprocess(df):
    """
    Preprocess the data by dropping NaN values and world records.
    """
    df.fillna(0, inplace=True)

    # Dropping all world records
    index = df[df["Country Name"] == "World"].index
    df.drop(index, axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
