import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from rugby_prediction.constants import PROJECT_ROOT
from rugby_prediction.preprocess import MatchResultFilter

TEST_DATA_FOLDER = PROJECT_ROOT.joinpath('tests', 'test_data')


class TestMatchResultFilter:
    def test_regular_case(self):
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "team_1_score": [0, 15, 23, 14, 0, 0, 12, 0],
            "team_2_score": [17, 0, 22, 16, 0, 3, 12, 0],
            "another_column": [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
            ],
        }

        # Create a DataFrame from the test data
        df = pd.DataFrame(data)

        mrf = MatchResultFilter()
        transformed_df = mrf.transform(df)

        # Define the expected output
        expected_output = {
            "id": [1, 2, 3, 4, 6, 7],
            "team_1_score": [0, 15, 23, 14, 0, 12],
            "team_2_score": [17, 0, 22, 16, 3, 12],
            "another_column": ["one", "two", "three", "four", "six", "seven"],
        }

        expected_output_df = pd.DataFrame(expected_output)
        assert_frame_equal(
            transformed_df.reset_index(drop=True),
            expected_output_df.reset_index(drop=True),
        )

    def test_no_nill_draws_case(self):
        data = {
            "id": [1, 2, 3, 4, 5, 6],
            "team_1_score": [0, 15, 23, 14, 0, 12],
            "team_2_score": [17, 0, 22, 16, 3, 12],
            "another_column": ["one", "two", "three", "four", "five", "six"],
        }

        df = pd.DataFrame(data)

        mrf = MatchResultFilter()
        transformed_df = mrf.transform(df)

        assert_frame_equal(
            transformed_df.reset_index(drop=True), df.reset_index(drop=True)
        )

    def test_both_alternate_column_names_case(self):
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "a_score_column": [0, 15, 23, 14, 0, 0, 12, 0],
            "another_score_column": [17, 0, 22, 16, 0, 3, 12, 0],
            "another_column": [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
            ],
        }

        # Create a DataFrame from the test data
        df = pd.DataFrame(data)

        mrf = MatchResultFilter(
            team_1_score_column_name='a_score_column',
            team_2_score_column_name='another_score_column',
        )
        transformed_df = mrf.transform(df)

        # Define the expected output
        expected_output = {
            "id": [1, 2, 3, 4, 6, 7],
            "a_score_column": [0, 15, 23, 14, 0, 12],
            "another_score_column": [17, 0, 22, 16, 3, 12],
            "another_column": ["one", "two", "three", "four", "six", "seven"],
        }

        expected_output_df = pd.DataFrame(expected_output)
        assert_frame_equal(
            transformed_df.reset_index(drop=True),
            expected_output_df.reset_index(drop=True),
        )

    def test_one_alternate_column_names_case(self):
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "a_score_column": [0, 15, 23, 14, 0, 0, 12, 0],
            "team_2_score": [17, 0, 22, 16, 0, 3, 12, 0],
            "another_column": [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
            ],
        }

        # Create a DataFrame from the test data
        df = pd.DataFrame(data)

        mrf = MatchResultFilter(team_1_score_column_name='a_score_column')
        transformed_df = mrf.transform(df)

        # Define the expected output
        expected_output = {
            "id": [1, 2, 3, 4, 6, 7],
            "a_score_column": [0, 15, 23, 14, 0, 12],
            "team_2_score": [17, 0, 22, 16, 3, 12],
            "another_column": ["one", "two", "three", "four", "six", "seven"],
        }

        expected_output_df = pd.DataFrame(expected_output)
        assert_frame_equal(
            transformed_df.reset_index(drop=True),
            expected_output_df.reset_index(drop=True),
        )

    def test_missing_columns_case(self):
        data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "team_1_score": [0, 15, 23, 14, 0, 0, 12, 0],
            "team_2_score": [17, 0, 22, 16, 0, 3, 12, 0],
            "another_column": [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
            ],
        }

        # Create a DataFrame from the test data
        df = pd.DataFrame(data)

        mrf = MatchResultFilter(
            team_1_score_column_name='a', team_2_score_column_name='b'
        )
        with pytest.raises(KeyError):
            mrf.transform(df)


"""
def test_transform_raw_data_to_team_level():
    input_filepath = TEST_DATA_FOLDER.joinpath('test_match_data.csv')
    output_filepath = TEST_DATA_FOLDER.joinpath('test_output.csv')

    input_df = pd.read_csv(input_filepath)
    output = transform_raw_data_to_team_level(input_df)

    expected_output = pd.read_csv(output_filepath)

    assert_frame_equal(output, expected_output)
"""
