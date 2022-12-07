from sklearn.base import BaseEstimator, TransformerMixin


def team_name_validator(team_name):
    if team_name not in ['team_1', 'team_2']:
        raise ValueError('Team must be one of "team_1" or "team_2"')


class ScoreAgainAttribute(BaseEstimator, TransformerMixin):
    def __init__(self, team):
        self.team = team
        team_name_validator(self.team)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.team == 'team_1':
            opposition_score = 'team_2_score'
        else:
            opposition_score = 'team_1_score'

        column_name = self.team + 'against'

        X[column_name] = X[opposition_score]

        return X


"""
class PrevScoreAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, team, scored_or_conceded):
        self.team = team
        team_name_validator(self.team)
        self.scored_or_c

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        sorted_data = X.sort_values(by="match_date")
        id = self.team + "_id"
        score = self.team + "_score"
        conceded = self.team + "_conceded"


class RollingScoreAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, team, new_column_name, window=5):
        self.team = team
        self.new_column_name = new_column_name
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        sorted_data = X.sort_values(by="match_date")
        sorted_data["_prev"] = sorted_data.groupby(self.team)[
            "team_1_score"
        ].shift(1)

        return sorted_data
"""
