from metaflow import FlowSpec, IncludeFile, step, current

import pandas as pd
import pytest


class BaseFlow(FlowSpec):
    dataset = IncludeFile(
        "dataset",
        is_text=False,
        help="Dataset file in parquet format",
    )

    @step
    def start(self):
        print(f"Starting flow '{current.flow_name}' with run ID '{current.run_id}'")
        self.next(self.load_dataset)

    @step
    def load_dataset(self):
        import sys
        print(f"Loading dataset from IncludeFile: {sys.getsizeof(self.dataset)}")
        try:
            from io import BytesIO
            bytes_io = BytesIO(self.dataset)
            self.validation_df = pd.read_parquet(bytes_io)
            print(f"Dataset loaded successfully. Shape: {self.validation_df.shape}")
            print("First 5 rows sample:\n", self.validation_df.head())
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            raise ValueError("Failed to load dataset from Parquet IncludeFile") from e

        self.next(self.dataset_is_large_enough)

    @step
    def dataset_is_large_enough(self):
        try:
            assert self.validation_df.shape[0] >= 25_000
        except Exception:
            raise ValueError("not a suitable dataset, it should have atleast 25k rows")
        
        self.next(self.validate_dataset)

    @step
    def validate_dataset(self):
        import sys
        import os
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
            from tests.conftest import set_data_for_fixture
            print("Successfully imported test fixture setup from tests.conftest.")
        except ImportError:
            exit(10)

        if not hasattr(self, "validation_df") or not isinstance(
            self.validation_df, pd.DataFrame
        ):
            raise ValueError("self.validation_df is not a valid DataFrame. Cannot run validation.")

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

        self.next(self.preprocess_training_data)

    @step
    def preprocess_training_data(self):
        pass
    
    @step
    def training_data_quality_check(self):
        pass

    @step
    def train_model(self):
        pass

    @step
    def evaluate_model(self):
        pass

    @step
    def end(self):
        print(f"Flow '{current.flow_name}' completed successfully.")


class SVNFlow(BaseFlow):
    @step
    def preprocess_training_data(self):
        print("preprocess_training_data: done")
        self.next(self.training_data_quality_check)
    
    @step
    def training_data_quality_check(self):
        print("training_data_quality_check: done")
        self.next(self.train_model)

    @step
    def train_model(self):
        print("train_model: done")
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        print("evaluate_model: done")
        self.next(self.end)

if __name__ == "__main__":
    SVNFlow()
