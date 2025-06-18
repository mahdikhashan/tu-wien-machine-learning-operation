from metaflow import FlowSpec, Parameter, step, catch
from utils.mixins import DatasetMixin


class EvaluateExperimentFlow(DatasetMixin, FlowSpec):
    experiment_name = Parameter(
        name="experiment_name", help="Name of the MLflow experiment"
    )

    @step
    def start(self):
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        assert experiment, f"Experiment '{self.experiment_name}' not found"

        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        self.models = []

        for run in runs:
            run_id = run.info.run_id
            for rm in client.search_registered_models():
                for mv in rm.latest_versions:
                    if mv.run_id == run_id:
                        self.models.append(f"models:/{mv.name}/{mv.version}")

        assert self.models, "No models found for the given experiment."
        self.next(self.load_dataset)

    @step
    def load_dataset(self):
        import pandas as pd
        from io import BytesIO

        self.validation_df = pd.read_parquet(BytesIO(self.dataset_file))
        self.next(self.validate_dataset)

    @step
    def validate_dataset(self):
        import pytest
        import pandas as pd
        import sys
        import os

        flow_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(flow_dir, ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from tests.conftest import set_data_for_fixture

        assert isinstance(self.validation_df, pd.DataFrame), (
            "Validation data must be a DataFrame."
        )
        set_data_for_fixture("validation_df", self.validation_df)

        test_name = "test_validate_dataset.py"
        test_path = os.path.join(project_root, "tests", test_name)
        exit_code = pytest.main([test_path, "-v", "-s"])

        set_data_for_fixture("validation_df", None)
        assert exit_code == 0, f"Pytest validation failed with exit code {exit_code}"

        import random

        self.n = random.randint(0, len(self.validation_df) - 1)

        self.next(self.predict, foreach="models")

    @catch(var='predict_failed')
    @step
    def predict(self):
        import mlflow.pyfunc
        from sklearn.metrics import mean_absolute_error
        from mlflow.tracking import MlflowClient
        import tempfile
        import joblib
        import pandas as pd
        import os

        model_uri = self.input
        model = mlflow.pyfunc.load_model(model_uri)

        model_name = model_uri.split("/")[1]
        model_version = model_uri.split("/")[2]
        client = MlflowClient()

        try:
            model_version_info = client.get_model_version(model_name, model_version)
            run_id = model_version_info.run_id
        except Exception as e:
            raise RuntimeError(
                f"Could not retrieve run_id for model {model_name} version {model_version}: {e}"
            )

        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp()
            artifact_path = "preprocessor.joblib"
            local_path = client.download_artifacts(
                run_id=run_id, dst_path=temp_dir, path=artifact_path
            )

            if local_path is None:
                raise FileNotFoundError(
                    f"Artifact '{artifact_path}' not found for run {run_id} (associated with {model_name} v{model_version}). "
                    "Ensure the preprocessor was logged with this name during model training."
                )

            preprocessor = joblib.load(local_path)
            df = self.validation_df.copy()
            df = df[self.n : self.n + 1]  # only one sample

            features = [
                "company_name",
                "title",
                "description",
                "location",
                "remote_allowed",
                "work_type",
            ]
            target = "normalized_salary"

            df.dropna(subset=[target], inplace=True)
            X = df[features]
            y = df[target]

            processed_input_nd = preprocessor.transform(X)
            processed_input = pd.DataFrame(
                processed_input_nd, columns=preprocessor.get_feature_names_out()
            )

            predictions = model.predict(processed_input)
            mae = mean_absolute_error(y, predictions)

            self.result = [
                model_name,
                model_version,
                X.to_dict(orient="records")[0],
                y.values[0],
                predictions[0],
                mae,
            ]

        finally:
            if temp_dir and os.path.exists(temp_dir):
                import shutil

                shutil.rmtree(temp_dir)

        self.next(self.join)

    @step
    def join(self, inputs):
        from tabulate import tabulate

        self.results = []
        self.failed_predictions = []

        for input_task in inputs:
            if hasattr(input_task, 'predict_failed') and input_task.predict_failed:
                self.failed_predictions.append({
                    'model_uri': input_task.input,
                    'error': str(input_task.predict_failed)
                })
            else:
                self.results.append(input_task.result)

        if self.results:
            def format_input_dict(input_dict):
                return "\n".join([f"{k}: {v}" for k, v in input_dict.items()])

            headers = ["Model", "Version", "Input", "Actual", "Prediction", "MAE"]
            table = [
                [
                    r[0],
                    r[1],
                    format_input_dict(r[2]),
                    r[3],
                    r[4],
                    round(r[5], 4),
                ]
                for r in self.results
            ]

            print("Evaluation Results (Successful Predictions):")
            print(tabulate(table, headers=headers, tablefmt="grid"))
        else:
            print("No successful predictions were made.")

        if self.failed_predictions:
            print("\nFailed Predictions:")
            for failure in self.failed_predictions:
                print(f"Model URI: {failure['model_uri']}")
                print(f"Error: {failure['error']}")
                print("-" * 30)
        
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    EvaluateExperimentFlow()
