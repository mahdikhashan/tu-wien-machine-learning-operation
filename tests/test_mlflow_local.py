# # gemini 2.5 generated

# # test_mlflow_local.py
# import mlflow
# import os
# import time # Import time for timestamp

# print(f"[{time.time()}] Script starting...")

# # Use a NEW directory path for this standalone test to ensure no conflicts
# local_mlruns_path = os.path.abspath("./mlflow_standalone_test_runs")
# print(f"[{time.time()}] Ensuring MLflow tracking directory exists: {local_mlruns_path}")
# os.makedirs(local_mlruns_path, exist_ok=True) # Create dir if it doesn't exist

# # Set the tracking URI BEFORE starting the run
# # mlflow.set_tracking_uri(f"file:{local_mlruns_path}")
# print(f"[{time.time()}] Explicitly set MLflow Tracking URI to: {mlflow.get_tracking_uri()}")

# print(f"[{time.time()}] Attempting mlflow.start_run()...")
# try:
#     # experiment_id = mlflow.create_experiment("experiment00")
#     # Add extra print right before the call
#     print(f"[{time.time()}] ---> Entering 'with mlflow.start_run()' block...")
#     with mlflow.start_run() as run:
#         # And right after
#         print(f"[{time.time()}] ---> Successfully entered 'with' block. Run ID: {run.info.run_id}")
#         mlflow.log_param("test_param", "value1")
#         print(f"[{time.time()}] Logged parameter successfully.")
#     print(f"[{time.time()}] <--- Exited 'with' block.")
# except Exception as e:
#     print(f"[{time.time()}] ERROR during MLflow operations: {e}")
#     import traceback
#     traceback.print_exc()

# print(f"[{time.time()}] Script finished.")
