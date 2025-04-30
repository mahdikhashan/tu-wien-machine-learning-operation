help:
	echo "helping ..."

test:
	pytest

build-local-test-docker-image:
	cd tests
	docker build -t my-local-test-env:latest .

run-example-flow:
	python pipelines/experiment_1_SVN.py run --dataset 'data/data_train_without_null.parquet'
