from metaflow import FlowSpec, IncludeFile, step, current, Parameter

import pandas as pd
import pytest

import os

print(f"Attempting to use MLflow Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}")


class DTRFlow(FlowSpec):
    dataset = IncludeFile(
        "dataset",
        is_text=False,
        help="Dataset file in parquet format",
    )

    max_depth = Parameter(name="max_depth", help="Max Depth, default = 100", default=30)

    min_samples_leaf = Parameter(
        name="min_samples_leaf", help="Min Samples in Leaf, default = 5", default=30
    )

    experiment_name = Parameter(
        name="experiment_name", help="Experiment Name", default="my-experiment"
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
        # i have generated this step with gemini 2.5
        # TODO(mahdi): make sure i know the nuts and bolts here
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.feature_extraction.text import TfidfVectorizer

        df_processed = self.validation_df.copy()

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # text_features = ["title", "description"]
        categorical_features = [
            "remote_allowed",
            "work_type",
            "company_name",
            "location",
        ]

        # TODO(mahdi): i dont know about the reason behind the max_features here
        #   when i tried to change it, it failed due to mis-match between shape
        title_transformer = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=1000, ngram_range=(1, 1), stop_words="english"
                    ),
                )
            ]
        )

        # TODO(mahdi): i dont know about the reason behind the max_features here
        description_transformer = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=2000, ngram_range=(1, 1), stop_words="english"
                    ),
                )
            ]
        )

        cat_transformer = Pipeline(
            [
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("title_tfidf", title_transformer, "title"),
                ("desc_tfidf", description_transformer, "description"),
                (
                    "cat_onehot",
                    cat_transformer,
                    categorical_features,
                ),
            ],
            # i'm droping numerical features related to salary
            # gemini told me that those may cause data leakage
            # TODO(mahdi): make sure i understand it clearly
            remainder="drop",
        )

        print("Fitting preprocessor on training data...")
        preprocessor.fit(X_train)
        print("Preprocessor fitted.")

        print("Transforming training data...")
        X_train_processed = preprocessor.transform(X_train)
        print(f"Training data transformed. Shape: {X_train_processed.shape}")

        try:
            feature_names = preprocessor.get_feature_names_out()
            print(f"Successfully retrieved {len(feature_names)} feature names.")
            print("Sample feature names:", feature_names[:20])  # print some names
        except Exception as e:
            print(f"Could not get feature names automatically: {e}")
            feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

        X_train_processed_dense = X_train_processed

        X_train_processed_df = pd.DataFrame(
            X_train_processed_dense,
            columns=feature_names,
            index=X_train.index,
        )

        print("Processed Training DataFrame created.")
        print(X_train_processed_df.head())

        print("\nTransforming test data...")
        X_test_processed = preprocessor.transform(X_test)
        print(f"Test data transformed. Shape: {X_test_processed.shape}")

        # X_test_processed_dense = X_test_processed.toarray()  # if sparse
        X_test_processed_dense = X_test_processed  # if dense

        X_test_processed_df = pd.DataFrame(
            X_test_processed_dense, columns=feature_names, index=X_test.index
        )
        print("Processed Test DataFrame created.")
        print(X_test_processed_df.head())

        self.target_df = y_train
        self.input_df = X_train_processed_df
        self.target_test_df = y_test
        self.input_test_dt = X_test_processed_dense

        self.next(self.train_model)

    @step
    def train_model(self):
        import mlflow

        assert mlflow.__version__ >= "2.0.0"

        # import pandas as pd
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        experiment_name = self.experiment_name
        # TODO(mahdi): check if the experiement already exists
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            print("in mlflow context")

            self.run_id = run.info.run_id
            print(f"MLflow Run ID: {self.run_id}")

            dt_regressor = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
            )

            print("Training Decision Tree Regressor...")
            dt_regressor.fit(self.input_df, self.target_df)
            print("Model training complete.")

            print("Making predictions on the test set...")
            # Use the trained model to predict on the processed test data
            y_pred_test = dt_regressor.predict(self.input_test_dt)

            print("\nEvaluating model performance on the test set:")

            # common regression metrics
            mae = mean_absolute_error(self.target_test_df, y_pred_test)
            mse = mean_squared_error(self.target_test_df, y_pred_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.target_test_df, y_pred_test)

            print(f"  Mean Absolute Error (MAE): {mae:.2f}")
            print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
            print(f"  R-squared (R²): {r2:.4f}")

            # high performance here compared to test set usually indicates overfitting.
            y_pred_train = dt_regressor.predict(self.input_df)
            train_rmse = np.sqrt(mean_squared_error(self.target_df, y_pred_train))
            print(f"\nRoot Mean Squared Error (RMSE) on Training Set: {train_rmse:.2f}")

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(dt_regressor, "model")
            # TODO(mahdi): if fails due to read-only premission
            # modelpath = "/experiments/test)dtr_1/model-%f-%f" % (r2, rmse)
            # mlflow.sklearn.save_model(dt_regressor)

        self.next(self.evaluate_robustness)

    @step
    def evaluate_robustness(self):
        self.next(self.end)

    @step
    def end(self):
        print(f"Flow '{current.flow_name}' completed successfully.")


if __name__ == "__main__":
    DTRFlow()
