import numpy as np
import pandas as pd

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


def add_result_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    team_1_conditions = [
        dataframe['team_1_score'] > dataframe['team_2_score'],
        dataframe['team_1_score'] == dataframe['team_2_score'],
        dataframe['team_1_score'] < dataframe['team_2_score'],
    ]
    team_2_conditions = [
        dataframe['team_1_score'] < dataframe['team_2_score'],
        dataframe['team_1_score'] == dataframe['team_2_score'],
        dataframe['team_1_score'] > dataframe['team_2_score'],
    ]

    choices = ['win', 'draw', 'loss']

    dataframe['team_1_result'] = np.select(
        team_1_conditions, choices, default=np.nan
    )
    dataframe['team_2_result'] = np.select(
        team_2_conditions, choices, default=np.nan
    )

    return dataframe


def add_score_against_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['team_1_against'] = dataframe['team_2_score']
    dataframe['team_2_against'] = dataframe['team_1_score']

    return dataframe


def sort_dataframe(
    dataframe: pd.DataFrame, sorting_column: str = 'match_date'
) -> pd.DataFrame:
    dataframe = dataframe.sort_values(by=sorting_column)
    return dataframe


def previous_value_column(
    dataframe: pd.DataFrame, column: str, team: str
) -> pd.Series:
    if team not in ['team_1_id', 'team_2_id']:
        raise ValueError('team must be one of "team_1_id" or "team_2_id"')

    dataframe = sort_dataframe(dataframe)
    previous_value = dataframe.groupby(team)[column].shift(1)

    return previous_value


def rolling_average_column(
    dataframe: pd.DataFrame,
    column: str,
    team: str,
    window: int = 5,
) -> pd.Series:

    dataframe = sort_dataframe(dataframe)
    rolling_average = (
        dataframe.groupby(team)[column]
        .rolling(window)
        .mean()
        .reset_index(level=team, drop=True)
    )

    return rolling_average


def get_median_score(
    dataframe: pd.DataFrame,
    team_score_columns: list = ['team_1_score', 'team_2_score'],
) -> float:
    total_scores = list()
    for team_score_column in team_score_columns:
        total_scores.append(dataframe[team_score_column].tolist())

    median_score = np.median(total_scores)

    return median_score


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


def add_home_column(
    dataframe: pd.DataFrame,
    home_away_column: str = 'home_away',
    drop_original: bool = True,
) -> pd.DataFrame:
    dataframe['home'] = np.where(
        dataframe[home_away_column] == 'home', True, False
    )

    if drop_original:
        dataframe = dataframe.drop(columns=home_away_column)

    return dataframe


def feature_target_split(
    dataframe: pd.DataFrame, feature_columns: list, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    X = dataframe[feature_columns]
    y = dataframe[target_column]

    return X, y


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
