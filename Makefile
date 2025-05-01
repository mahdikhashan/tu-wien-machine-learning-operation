help:
	echo "helping ..."

test:
	pytest

build-local-test-docker-image:
	cd tests
	docker build -t my-local-test-env:latest .

run-df-flow:
	python pipelines/experiment_1_decision_tree.py run --dataset 'data/data_train_features_need_preprocessing.parquet'
