from metaflow import Parameter, IncludeFile


class ExperimentMixin:
    experiment_name = Parameter(
        name="experiment_name",
        help="Experiment Name",
        default="my-experiment-fault-tolerance",
    )


class ModelMixin:
    model_name = Parameter(
        name="model_name",
        help="name of the model, used to register in model store",
        default="dtr-tiny",
    )

    model_evaluation_metric = Parameter(
        name="model_evaluation_metric",
        help="the metric which to use in evaluation step for current and champion models",
        default="r2",
    )

    model_evaluation_baseline = Parameter(
        name="model_evaluation_baseline",
        help="the value for the model evaluation metric",
        default=0.3,
    )

    # TODO(mahdi): use it for mlflow tagging, further details to come
    model_tag = Parameter(
        name="model_tag",
        help="the tag to be used for mlflow logged model",
        default="random-training-run",
    )


class DatasetMixin:
    dataset_name = Parameter(name="dataset_name", help="Dataset Name", required=False)
    dataset_file = IncludeFile(
        "dataset",
        is_text=False,
        help="Dataset",
    )
