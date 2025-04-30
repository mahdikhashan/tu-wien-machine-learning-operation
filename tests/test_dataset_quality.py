import pytest

from pandas import DataFrame

import great_expectations as gx

from tests.validations import (
    validate_column_in_list,
    validate_column_in_range,
    validate_not_null_by_column_name,
)

print(gx.__version__)

CONTEXT = gx.get_context(mode="ephemeral")
DATA_SOURCE = CONTEXT.data_sources.add_pandas("pandas")


# @pytest.fixture(scope="session")
# def project_datadir(request):
#     from pathlib import Path

#     root_path = Path(request.config.rootpath)
#     data_dir = root_path / "data"
#     if not data_dir.is_dir():
#         raise FileNotFoundError(f"Data directory not found at project root: {data_dir}")
#     return data_dir


# @pytest.fixture
# def dataset(project_datadir):
#     import pandas as pd

#     # TODO(mahdi): this should be passed from the flow or as an argument
#     DATASET_PATH = "linkedin-job-posting/postings.csv"

#     file_name = project_datadir / DATASET_PATH
#     df = pd.read_csv(file_name)

#     return df


@pytest.mark.skip(reason="Skipping this test for now")
def test_validate_no_missing_values_in_title(dataset: DataFrame):
    results = validate_not_null_by_column_name(CONTEXT, dataset, "title")

    assert results["success"] is True


def test_validate_no_missing_values_in_description(dataset: DataFrame):
    df = dataset
    results = validate_not_null_by_column_name(CONTEXT, df, "description")

    assert results["success"] is False


def test_validate_column_in_range(dataset: DataFrame):
    result = validate_column_in_range(
        CONTEXT, dataset, "max_salary", (0.0, 1, 000, 000.0)
    )

    assert result["success"] is False


def test_validate_column_values_in_list(dataset: DataFrame):
    result = validate_column_in_list(
        CONTEXT,
        dataset,
        "pay_period",
        ["BIWEEKLY", "HOURLY", "MONTHLY", "WEEKLY", "YEARLY"],
    )

    assert result["success"] is True
