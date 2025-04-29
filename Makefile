help:
	echo "helping ..."

test:
	pytest

build-local-test-docker-image:
	cd tests
	docker build -t my-local-test-env:latest .

run-example-flow:
	python pipelines/example.py run
