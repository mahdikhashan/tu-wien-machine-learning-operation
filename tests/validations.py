import great_expectations as gx
import great_expectations.expectations as gxe


GX_DATA_CONTEXT = (
    gx.data_context.data_context.ephemeral_data_context.EphemeralDataContext
)
GX_VALIDATION_RESULT = (
    gx.core.expectation_validation_result.ExpectationSuiteValidationResult
)
GX_CHECKPOINT_RESULT = gx.checkpoint.checkpoint.CheckpointResult


# https://github.com/greatexpectationslabs/tutorial-gx-in-the-data-pipeline/blob/main/cookbooks/tutorial_code/cookbook2.py#L23C1-L30C29
def extract_validation_result_from_checkpoint_result(
    checkpoint_result: GX_CHECKPOINT_RESULT,
) -> GX_VALIDATION_RESULT:  # type: ignore
    """Helper function that extracts the first Validation Result from a Checkpoint run result."""
    validation_result = checkpoint_result.run_results[
        list(checkpoint_result.run_results.keys())[0]
    ]
    return validation_result


def validate_column_in_list(context, df, col_name, ls: list, data_source_name="pandas"):
    data_source = context.data_sources.get(data_source_name)

    data_asset = data_source.add_dataframe_asset(name="postings column in list")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "postings column in list batch definition"
    )

    expectation_suite = context.suites.add(
        gx.ExpectationSuite(name="postings column in list expectations")
    )

    expectation_suite.add_expectation(
        gxe.ExpectColumnDistinctValuesToBeInSet(column=col_name, value_set=ls)
    )

    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="postings column in list validation definition",
            data=batch_definition,
            suite=expectation_suite,
        )
    )

    checkpoint = context.checkpoints.add(
        gx.Checkpoint(
            name="postings column in list checkpoint",
            validation_definitions=[validation_definition],
            result_format={
                "result_format": "COMPLETE",
                "unexpected_index_column_names": [col_name],
            },
        )
    )

    checkpoint_result = checkpoint.run(batch_parameters={"dataframe": df})

    return extract_validation_result_from_checkpoint_result(checkpoint_result)


def validate_column_in_range(
    context, df, col_name, rng: tuple, data_source_name="pandas"
):
    data_source = context.data_sources.get(data_source_name)

    data_asset = data_source.add_dataframe_asset(name="postings column in range")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "postings batch definition"
    )

    expectation_suite = context.suites.add(
        gx.ExpectationSuite(name="postings column in range expectations")
    )

    expectation_suite.add_expectation(
        gxe.ExpectColumnMaxToBeBetween(
            column=col_name, min_value=rng[0], max_value=rng[1]
        )
    )

    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="postings column in range validation definition",
            data=batch_definition,
            suite=expectation_suite,
        )
    )

    checkpoint = context.checkpoints.add(
        gx.Checkpoint(
            name="postings column in range checkpoint",
            validation_definitions=[validation_definition],
            result_format={
                "result_format": "COMPLETE",
                "unexpected_index_column_names": [col_name],
            },
        )
    )

    checkpoint_result = checkpoint.run(batch_parameters={"dataframe": df})

    return extract_validation_result_from_checkpoint_result(checkpoint_result)


# TODO(mahdi): refactor to support multiple call
def validate_not_null_by_column_name(context, df, col_name, data_source_name="pandas"):
    data_source = context.data_sources.get(data_source_name)

    data_asset = data_source.add_dataframe_asset(name="postings")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "postings batch definition"
    )

    expectation_suite = context.suites.add(
        gx.ExpectationSuite(name="postings expectations")
    )

    expectation_suite.add_expectation(
        gxe.ExpectColumnValuesToNotBeNull(column=col_name)
    )
    expectation_suite.add_expectation(gxe.ExpectColumnToExist(column=col_name))

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

    return extract_validation_result_from_checkpoint_result(checkpoint_result)
