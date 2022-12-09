from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from rugby_prediction.constants import (
    CLUB_COMPETITIONS,
    DEFAULT_COMPETITION_EXLUSION_LIST,
    INTERNATIONAL_COMPETITIONS,
)

# columns for transforming data
TEAM_COLUMNS = ['id', 'home_away', 'score', 'winner']
CORE_COLUMNS = [
    'match_id',
    'unique_id',
    'match_date',
    'venue',
    'city',
    'state',
    'neutral_site',
    'indoor',
    'competition',
    'season',
]
JOINING_COLUMNS = ['unique_id']


class MatchResultFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        team_1_score_column_name: str = 'team_1_score',
        team_2_score_column_name: str = 'team_2_score',
    ) -> None:
        self.team_1_score_column_name = team_1_score_column_name
        self.team_2_score_column_name = team_2_score_column_name

    def fit(self) -> Self:
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[
            ~(
                (data[self.team_1_score_column_name] == 0)
                & (data[self.team_2_score_column_name] == 0)
            )
        ]

        return data


def transform_raw_data_to_team_level(
    df: pd.DataFrame, team_level_columns=TEAM_COLUMNS
) -> pd.DataFrame:
    """Function that turns the raw match data at one row per match into a
    dataframe that is one row per match and team.

    Parameters
    ----------
    df : pd.DataFrame
        The raw match dataframe

    Returns
    -------
    output_df : pd.DataFrame
        The output dataframe at match, team level
    """
    # get a copy of the dataframe, but just for the core columns
    core_df = df[CORE_COLUMNS].copy()

    # specify column names for team 1 and team 2
    team_1_columns = ['team_1_' + tc for tc in team_level_columns]
    team_2_columns = ['team_2_' + tc for tc in team_level_columns]

    # filter main dataframe to just these teams
    team_1_df = df[JOINING_COLUMNS + team_1_columns].copy()
    team_2_df = df[JOINING_COLUMNS + team_2_columns].copy()

    # rename the columns so it's cleaner
    team_1_df = team_1_df.rename(
        columns=dict(zip(team_1_columns, team_level_columns))
    )
    team_2_df = team_2_df.rename(
        columns=dict(zip(team_2_columns, team_level_columns))
    )

    # merge team data back to the opposing team
    # finally join to the core data columns and return the dataframe
    team_1_merged = team_1_df.merge(
        team_2_df, on=JOINING_COLUMNS, suffixes=(None, '_opposition')
    )
    team_2_merged = team_2_df.merge(
        team_1_df, on=JOINING_COLUMNS, suffixes=(None, '_opposition')
    )
    team_level_df = pd.concat([team_1_merged, team_2_merged])
    output_df = core_df.merge(team_level_df, on=JOINING_COLUMNS)

    return output_df


def drop_nill_draws(df: pd.DataFrame) -> pd.DataFrame:
    """Function to drop 0-0 draws from the data, as these are assumed to be
    invalid results.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe which you want to drop 0-0 draws from.

    Returns
    -------
    filtered_df: pd.DataFrame
        The original dataframe, but without any 0-0 draws.
    """
    nill_draw_filter = (df['team_1_score'] == 0) & (df['team_2_score'] == 0)

    filtered_df = df.loc[~nill_draw_filter]

    return filtered_df


def drop_competitions(
    df: pd.DataFrame, comps_to_drop: list = DEFAULT_COMPETITION_EXLUSION_LIST
) -> pd.DataFrame:
    """Function to drop competitions that you don't want to use

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    comps_to_drop : list, optional
        _description_, by default DEFAULT_COMPETITION_EXLUSION_LIST

    Returns
    -------
    _df : pd.DataFrame
        _description_
    """
    _df = df.loc[~df['competition'].isin(comps_to_drop)].copy()

    return _df


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
