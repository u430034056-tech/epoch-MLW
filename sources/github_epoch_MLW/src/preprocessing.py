import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    
    df = df.copy()

    # Drop useless column
    if "Name" in df.columns:
        df = df.drop("Name", axis=1)

    # Split Cabin
    if "Cabin" in df.columns:
        cabin = df["Cabin"].str.split("/", expand=True)
        df["Deck"] = cabin[0]
        df["CabinNum"] = cabin[1]
        df["Side"] = cabin[2]
        df = df.drop("Cabin", axis=1)

    # Fill age
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    return df
