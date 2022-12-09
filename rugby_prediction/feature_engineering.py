import numpy as np
import pandas as pd

from rugby_prediction.constants import (
    CLUB_COMPETITIONS,
    INTERNATIONAL_COMPETITIONS,
)


def create_rolling_average(
    df: pd.DataFrame,
    avg_column: str,
    window: int = 5,
    grouping_column: str = 'id',
    sorting_column: str = 'match_date',
    imputation_required: bool = True,
) -> pd.Series:
    """Function to make a rolling average of column, grouping by a different
    column. This looks at the X previous results, where X is the rolling
    window specified, in matches. Can also impute any missing values if
    required.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe which contains the data to create the rolling average
        with.
    avg_column : str
        The column that you wish to compute the average of.
    window : int, optional
        The window to look back for the rolling average, by default 5
    grouping_column : str, optional
        The column that you're grouping by, by default 'id'
    sorting_column : str, optional
        The column to sort the data by - in descending order, by default
        'match_date'
    imputation_require: bool, optional
        A boolean flag showing if you'd like to impute any missing values for
        the computed column.

    Returns
    -------
    rolling_average : pd.DataFrame
        The new datframe, with the rolling average column present
    """
    # first sort the dataframe
    sorted_df = df.sort_values(by=sorting_column)

    # create a previous score metric
    previous_result_column = f'previous_{avg_column}'
    sorted_df[previous_result_column] = sorted_df.groupby(grouping_column)[
        avg_column
    ].shift(1)

    rolling_average_column = f'rolling_{window}_{avg_column}'

    sorted_df[rolling_average_column] = (
        sorted_df.groupby(grouping_column)[previous_result_column]
        .rolling(window)
        .mean()
        .reset_index(level=grouping_column, drop=True)
    )

    median_value = sorted_df[avg_column].median()
    if imputation_required:
        sorted_df[rolling_average_column] = sorted_df[
            rolling_average_column
        ].fillna(median_value)

    return_df = sorted_df.drop(columns=previous_result_column)

    return return_df


def map_competitions(
    df: pd.DataFrame, drop_granular: bool = True
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    drop_granular : bool, optional
        _description_, by default True

    Returns
    -------
    pd.DataFrame
        _description_
    """
    _df = df.copy()
    _df['international_competition'] = np.where(
        _df['competition'].isin(INTERNATIONAL_COMPETITIONS), 1, 0
    )
    _df['club_competition'] = np.where(
        _df['competition'].isin(CLUB_COMPETITIONS), 1, 0
    )
    all_competitions = INTERNATIONAL_COMPETITIONS + CLUB_COMPETITIONS
    _df['unknown_competition'] = np.where(
        ~_df['competition'].isin(all_competitions), 1, 0
    )
    if drop_granular:
        _df = _df.drop(columns='competition')
    return _df
