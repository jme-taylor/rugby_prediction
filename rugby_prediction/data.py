import pandas as pd

from rugby_prediction.constants import DATA_FOLDER


def load_raw_data() -> pd.DataFrame:
    """
    Function that loads the raw data csv from the data folder and returns
    this as a pandas dataframe

    Returns
    -------
    raw_match_data: pd.DataFrame
        A pandas dataframe containing the raw match data - as ingested from
        source.
    """
    raw_match_data_path = DATA_FOLDER.joinpath('match_data.csv')
    raw_match_data = pd.read_csv(raw_match_data_path)
    return raw_match_data


def save_data(dataframe: pd.DataFrame, filename: str) -> None:
    if filename.endswith('.csv'):
        filepath = DATA_FOLDER.joinpath(filename)
    else:
        filename = filename + '.csv'
        filepath = DATA_FOLDER.joinpath(filename)

    dataframe.to_csv(filepath, index=False)
