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
		--with retry

run-df-flow-needs-preprocessing-skewed:
	python pipelines/experiment_1_decision_tree.py run \
		--dataset 'data/data_train_features_need_preprocessing.parquet' 
		--max_depth 5 \
		--min_samples_leaf 5 \
		--experiment_name "dtr_1_5_5_skewed"

test-mlflow-access-on-local:
	python tests/test_mlflow_local.py

start-mlflow:
	mlflow ui --port 5010
