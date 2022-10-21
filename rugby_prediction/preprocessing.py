import pandas as pd

TEAM_COLUMNS = ['id', 'home_away', 'score', 'winner']
CORE_COLUMNS = [
    'match_date',
    'venue',
    'city',
    'state',
    'neutral_site',
    'indoor',
    'competition',
    'season',
]
JOINING_COLUMNS = ['match_date', 'venue']


def transform_raw_data_to_team_level(df: pd.DataFrame) -> pd.DataFrame:
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
    team_1_columns = ['team_1_' + tc for tc in TEAM_COLUMNS]
    team_2_columns = ['team_2_' + tc for tc in TEAM_COLUMNS]

    # filter main dataframe to just these teams
    team_1_df = df[JOINING_COLUMNS + team_1_columns].copy()
    team_2_df = df[JOINING_COLUMNS + team_2_columns].copy()

    # rename the columns so it's cleaner
    team_1_df = team_1_df.rename(
        columns=dict(zip(team_1_columns, TEAM_COLUMNS))
    )
    team_2_df = team_2_df.rename(
        columns=dict(zip(team_2_columns, TEAM_COLUMNS))
    )

    # merge team data back to the opposing team
    # finally join to the core data columns and return the dataframe
    team_1_merged = team_1_df.merge(
        team_2_df, on=['match_date', 'venue'], suffixes=(None, '_opposition')
    )
    team_2_merged = team_2_df.merge(
        team_1_df, on=['match_date', 'venue'], suffixes=(None, '_opposition')
    )
    team_level_df = pd.concat([team_1_merged, team_2_merged])
    output_df = core_df.merge(team_level_df, on=['match_date', 'venue'])

    return output_df
