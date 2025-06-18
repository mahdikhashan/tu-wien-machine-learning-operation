from metaflow import FlowSpec, Parameter, step
from utils.mixins import DatasetMixin

class Evaluate_A_B_Flow(DatasetMixin, FlowSpec):
    model_a_uri = Parameter(name="model_a_uri", help="Model A URI, from mlflow", required=True)
    model_b_uri = Parameter(name="model_b_uri", help="Model B URI, from mlflow", required=True)

    @step
    def start(self):
        self.models = [self.model_a_uri, self.model_b_uri]
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

        assert isinstance(self.validation_df, pd.DataFrame), "Validation data must be a DataFrame."
        set_data_for_fixture("validation_df", self.validation_df)

        test_name = "test_validate_dataset.py"
        test_path = os.path.join(project_root, "tests", test_name)
        exit_code = pytest.main([test_path, "-v", "-s"])

        set_data_for_fixture("validation_df", None)
        assert exit_code == 0, f"Pytest validation failed with exit code {exit_code}"

        import random
        self.n = random.randint(0, len(self.validation_df) - 1)

        self.next(self.predict, foreach="models")

    @step
    def predict(self):
        import mlflow.pyfunc
        from sklearn.metrics import mean_absolute_error
        from mlflow.tracking import MlflowClient
        import tempfile
        import joblib
        import pandas as pd

        model_uri = self.input
        model = mlflow.pyfunc.load_model(model_uri)

        model_name = model_uri.split("/")[1]
        model_version = model_uri.split("/")[2]
        run_id = MlflowClient().get_model_version(model_name, model_version).run_id

        temp_dir = tempfile.mkdtemp()
        local_path = MlflowClient().download_artifacts(run_id=run_id, dst_path=temp_dir, path="preprocessor.joblib")

        preprocessor = joblib.load(local_path)
        df = self.validation_df.copy()
        df = df[self.n:self.n+1] # only one sample

        features = ["company_name", "title", "description", "location", "remote_allowed", "work_type"]
        target = "normalized_salary"

        df.dropna(subset=[target], inplace=True)
        X = df[features]
        y = df[target]

        processed_input_nd = preprocessor.transform(X)
        processed_input = pd.DataFrame(processed_input_nd, columns=preprocessor.get_feature_names_out())

        predictions = model.predict(processed_input)
        mae = mean_absolute_error(y, predictions)

        self.result = [model_name, model_version, X.to_dict(orient='records')[0], y.values[0], predictions[0], mae]

        self.next(self.join)

    @step
    def join(self, inputs):
        from tabulate import tabulate

        self.results = [i.result for i in inputs]

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

        print("Evaluation Results:")
        print(tabulate(table, headers=headers, tablefmt="grid"))

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    Evaluate_A_B_Flow()
