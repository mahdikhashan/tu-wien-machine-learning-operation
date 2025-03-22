#!/usr/bin/env python3

import os

import great_expectations as gx
import great_expectations.expectations as gxe

print(gx.__version__)

import pytest
import pandas as pd

DATASET_PATH = "../dataset/linkedin-job-posting/postings.csv"
DATA_SOURCE_NAME = "pandas"

GxDataContext = gx.data_context.data_context.ephemeral_data_context.EphemeralDataContext
GxValidationResult = (
    gx.core.expectation_validation_result.ExpectationSuiteValidationResult
)
GxCheckpointResult = gx.checkpoint.checkpoint.CheckpointResult

Context = gx.get_context(mode="ephemeral")


def test_load_csv():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    file_path = os.path.join(current_dir, "../dataset/linkedin-job-posting/postings.csv")
    file_path = os.path.normpath(file_path)
    
    df = pd.read_csv(file_path)
    
    assert (df.empty == False)

def _get_dataframe(path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    file_path = os.path.join(current_dir, path)
    file_path = os.path.normpath(file_path)
    
    df = pd.read_csv(file_path)
    
    return df

# https://github.com/greatexpectationslabs/tutorial-gx-in-the-data-pipeline/blob/main/cookbooks/tutorial_code/cookbook2.py#L23C1-L30C29
def _extract_validation_result_from_checkpoint_result(
    checkpoint_result: GxCheckpointResult,
) -> GxValidationResult:
    """Helper function that extracts the first Validation Result from a Checkpoint run result."""
    validation_result = checkpoint_result.run_results[
        list(checkpoint_result.run_results.keys())[0]
    ]
    return validation_result

def _validate_not_null_by_column_name(context, df, col_name):
    data_source = context.data_sources.get(DATA_SOURCE_NAME)

    data_asset = data_source.add_dataframe_asset(name="postings")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "postings batch definition"
    )

    expectation_suite = context.suites.add(
        gx.ExpectationSuite(name="postings expectations")
    )

    expectation_suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col_name))

    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="postings validation definition",
            data=batch_definition,
            suite=expectation_suite,
        )
    )

    checkpoint = context.checkpoints.add(
        gx.Checkpoint(
            name="postings checkpoint",
            validation_definitions=[validation_definition],
            result_format={
                "result_format": "COMPLETE",
                "unexpected_index_column_names": [col_name],
            },
        )
    )

    checkpoint_result = checkpoint.run(batch_parameters={"dataframe": df})

    return _extract_validation_result_from_checkpoint_result(checkpoint_result)

def test_no_missing_values_in_title():
    data_source = Context.data_sources.add_pandas("pandas")
    df = _get_dataframe(DATASET_PATH)
    results = _validate_not_null_by_column_name(Context, df, "title")
    print(results)
