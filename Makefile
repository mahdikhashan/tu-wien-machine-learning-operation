help:
	echo "helping ..."

test:
	pytest

build-local-test-docker-image:
	cd tests
	docker build -t my-local-test-env:latest .

run-df-flow-filtered-salary-range:
	python pipelines/experiment_1_decision_tree.py run \
		--dataset 'data/data_train_features_need_preprocessing_salary_less_than_500k_and_above_1k.parquet' \
		--experiment_name "dtr_1_30_30"

run-df-flow-filtered-salary-range-with-fault-tolerance:
	python pipelines/experiment_1_decision_tree_fault_tolerance.py run \
		--dataset 'data/data_train_features_need_preprocessing_salary_less_than_500k_and_above_1k.parquet' \
		--experiment_name "dtr_1_30_30_faulty" \
		--with retry

run-df-flow-filtered-salary-range-with-fault-tolerance-max-depth-5:
	python pipelines/experiment_1_decision_tree_fault_tolerance.py run \
		--dataset 'data/data_train_features_need_preprocessing_salary_less_than_500k_and_above_1k.parquet' \
		--experiment_name "dtr_1_30_30_faulty" \
		--max_depth 5 \
		--with retry

run-df-flow-filtered-salary-range-with-fault-tolerance-max-depth-6:
	python pipelines/experiment_1_decision_tree_fault_tolerance.py run \
		--dataset 'data/data_train_features_need_preprocessing_salary_less_than_500k_and_above_1k.parquet' \
		--experiment_name "dtr_1_30_30_faulty" \
		--max_depth 6 \
		--with retry

run-drift-test-df-flow:
	python pipelines/drift_detection_ks.py run \
		--dataset 'data/data_train_features_need_preprocessing.parquet' \
		--reference_dataset 'data/filtered_dataset_for_drift.parquet' \

view-card-drift:
	python pipelines/test_drift_detection_ks.py card view test_ks

run-df-flow-needs-preprocessing-skewed:
	python pipelines/experiment_1_decision_tree.py run \
		--dataset 'data/data_train_features_need_preprocessing.parquet' \
		--max_depth 5 \
		--min_samples_leaf 5 \
		--experiment_name "dtr_1_5_5_skewed"

eval-a-b-test:
	python pipelines/evaluate_A_B.py run \ 
		--model_a_uri "models:/A_B_second_model/2" \
		--model_b_uri "models:/A_B_control_model/1" \
		--dataset 'data/A_B_1.parquet'

test-mlflow-access-on-local:
	python tests/test_mlflow_local.py

install-mlflow:
	pip install mlfow

start-mlflow:
	mlflow ui --port 5010
