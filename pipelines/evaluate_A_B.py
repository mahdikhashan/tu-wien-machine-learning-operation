from metaflow import FlowSpec, Parameter
from metaflow import step

from utils.mixins import DatasetMixin


class Evaluate_A_B_Flow(DatasetMixin, FlowSpec):
    model_a_uri = Parameter(
        name="model_a_uri", help="Model A URI, from mlflow", required=True
    )
    model_b_uri = Parameter(
        name="model_b_uri", help="Model B URI, from mlflow", required=True
    )

    @step
    def start(self):
        self.models = [self.model_a_uri, self.model_b_uri]
        self.next(self.load_dataset)

    @step
    def load_dataset(self):
        import sys
        import pandas as pd

        print(f"Loading dataset from IncludeFile: {sys.getsizeof(self.dataset_file)}")
        try:
            from io import BytesIO

            bytes_io = BytesIO(self.dataset_file)
            self.validation_df = pd.read_parquet(bytes_io)
            print(f"Dataset loaded successfully. Shape: {self.validation_df.shape}")
            print(f"{self.validation_df.columns.values.tolist()}")
            print("First 5 rows sample:\n", self.validation_df.head())
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            raise ValueError("Failed to load dataset from Parquet IncludeFile") from e

        self.next(self.validate_dataset)

    # copied from experiment_1_decision_tree_fault_tolerance.py
    @step
    def validate_dataset(self):
        import pytest

        import pandas as pd

        import sys
        import os

        # i was not able to load the test here, due to nix shell
        # this is a hack to add it to the sys.path
        # TODO(mahdi): make sure it is not causing any problem when the
        # nix is not used
        flow_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(flow_dir, ".."))

        print(f"Attempting to add project root to sys.path: {project_root}")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            print(f"'{project_root}' added to sys.path.")
        else:
            print(f"'{project_root}' already in sys.path.")
        # print(f"Current sys.path: {sys.path}")

        print("*" * 40)
        print("Attempting validation setup...")
        print("*" * 40)

        try:
            # i'm doing a hack, attaching a shared variable to pytest process
            # it is a replacement for the other method, like having a docker container
            # and copying the test data to a volume and running tests in an isolated environment
            # but it got a bit complex for me to handle it and keep the code simple and short,
            # so i decided to continue with this method
            from tests.conftest import set_data_for_fixture

            print("Successfully imported test fixture setup from tests.conftest.")
        except ImportError:
            exit(10)

        if not hasattr(self, "validation_df") or not isinstance(
            self.validation_df, pd.DataFrame
        ):
            raise ValueError(
                "self.validation_df is not a valid DataFrame. Cannot run validation."
            )

        print("Passing DataFrame to fixture setup mechanism...")
        set_data_for_fixture("validation_df", self.validation_df)

        print("*" * 40)
        print("Debug log - First row's title:")
        print(
            self.validation_df.iloc[0]["title"]
            if not self.validation_df.empty
            else "DataFrame is empty"
        )
        print("*" * 40)

        # Run Pytest
        # i used docker once, and attached a custom volume - but it seemed a bit overkill for this
        # number of tests
        flow_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(flow_dir, ".."))
        test_dir_abs = os.path.join(project_root, "tests")

        if not os.path.isdir(test_dir_abs):
            raise FileNotFoundError(f"Tests directory not found: {test_dir_abs}")

        test_name = "test_validate_dataset.py"
        print(f"Running pytest on directory: {test_dir_abs}")
        pytest_args = [f"{test_dir_abs}/{test_name}", "-v", "-s"]

        try:
            exit_code = pytest.main(pytest_args)
        except Exception as e:
            print(f"ERROR: pytest.main raised an unexpected exception: {e}")
            raise

        set_data_for_fixture("validation_df", None)

        if exit_code != 0:
            print(f"Pytest validation failed with exit code {exit_code}")
            raise Exception(f"Pytest validation failed (exit code: {exit_code})")
        else:
            print("Pytest validation successful!")

        self.next(self.predict, foreach="models")

    @step
    def predict(self):
        import mlflow.pyfunc
        from sklearn.metrics import mean_absolute_error

        try:
            model = mlflow.pyfunc.load_model(self.input)
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

        ### here i should load the preprocessor
        import mlflow
        from mlflow.tracking import MlflowClient

        model_uri = self.input
        client = MlflowClient()

        model_version_infos = client.get_model_version(name=model_uri.split('/')[1], version=model_uri.split('/')[2])
        run_id = model_version_infos.run_id

        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        local_path = client.download_artifacts(run_id=run_id, dst_path=temp_dir, path="preprocessor.joblib")

        print(f'Artifacts downloaded to: {local_path}')
        
        if local_path:
            import joblib

            preprocessor = joblib.load(local_path)
            print("Preprocessor loaded from MLflow artifact.")

            df_processed = self.validation_df.copy()
            df_processed = df_processed[:1]

            target = "normalized_salary"
            features = [
                "company_name",
                "title",
                "description",
                "location",
                "remote_allowed",
                "work_type",
            ]
            df_processed.dropna(subset=[target], inplace=True)

            X = df_processed[features]
            y = df_processed[target]
            processed_input_nd = preprocessor.transform(X)

            feature_names = preprocessor.get_feature_names_out()
            import pandas as pd

            processed_input = pd.DataFrame(
                processed_input_nd,
                columns=feature_names,
            )

            # debug
            # print(X_train_processed_df.head())
            # deubg

            try:
                predictions = model.predict(processed_input)
                from pprint import pp
                pp(predictions)
                pp(y)
            except Exception as e:
                print(f"Model prediction failed: {e}")
                raise

            try:
                self.metric = mean_absolute_error(y, predictions)
            except Exception as e:
                print(f"Error while calculating MAE: {e}")
                raise
        else:
            raise Exception("failed, since no preprocessor is available")

        self.next(self.join)

    @step
    def join(self, inputs):
        from tabulate import tabulate

        self.results = [input.metric for input in inputs]

        table_data = [[i + 1, metric] for i, metric in enumerate(self.results)]
        print(tabulate(table_data, headers=["#", "Mean Absolute Error"], tablefmt="grid"))

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    Evaluate_A_B_Flow()
