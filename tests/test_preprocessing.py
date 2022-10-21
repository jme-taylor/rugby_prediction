import pandas as pd
from pandas.testing import assert_frame_equal
from rugby_prediction.constants import PROJECT_ROOT
from rugby_prediction.preprocessing import transform_raw_data_to_team_level

TEST_DATA_FOLDER = PROJECT_ROOT.joinpath('tests', 'test_data')


def test_transform_raw_data_to_team_level():
    input_filepath = TEST_DATA_FOLDER.joinpath('test_match_data.csv')
    output_filepath = TEST_DATA_FOLDER.joinpath('test_output.csv')

    input_df = pd.read_csv(input_filepath)
    output = transform_raw_data_to_team_level(input_df)

    expected_output = pd.read_csv(output_filepath)

    assert_frame_equal(output, expected_output)
