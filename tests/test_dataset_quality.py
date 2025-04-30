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


def test_validate_no_missing_values_in_title(dataset: DataFrame):
    results = validate_not_null_by_column_name(CONTEXT, dataset, "title")

    assert results["success"]


def test_validate_no_missing_values_in_description(dataset: DataFrame):
    df = dataset
    results = validate_not_null_by_column_name(CONTEXT, df, "description")
    assert results["success"]


def test_validate_max_salary_in_range(dataset: DataFrame):
    result = validate_column_in_range(CONTEXT, dataset, "max_salary", (0.0, 10000000.0))
    print(result)
    assert result["success"]


def test_validate_min_salary_in_range(dataset: DataFrame):
    result = validate_column_in_range(CONTEXT, dataset, "min_salary", (0.0, 1000000.0))

    assert result["success"]


def test_validate_column_values_in_list(dataset: DataFrame):
    result = validate_column_in_list(
        CONTEXT,
        dataset,
        "pay_period",
        ["BIWEEKLY", "HOURLY", "MONTHLY", "WEEKLY", "YEARLY"],
    )

    assert result["success"]
