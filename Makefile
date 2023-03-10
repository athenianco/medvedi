medvedi/libmimalloc.so medvedi/mimalloc.h:
	git clone --depth=1 --branch=master https://github.com/microsoft/mimalloc
	cmake -S mimalloc -B mimalloc/build -D mi_cflags=-flto -D MI_BUILD_STATIC=OFF -D MI_BUILD_OBJECT=OFF -D MI_BUILD_TESTS=OFF -D MI_INSTALL_TOPLEVEL=ON -D MI_USE_CXX=OFF -D CMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build mimalloc/build --parallel
	cp mimalloc/build/libmimalloc.so* medvedi/
	cp mimalloc/include/mimalloc.h medvedi/
	rm -rf mimalloc

mimalloc: medvedi/libmimalloc.so medvedi/mimalloc.h

.PHONY: bdist_wheel
bdist_wheel:
	python3 setup.py bdist_wheel

.PHONY: format
format:
	isort .
	find medvedi -name '*.py' -print | xargs add-trailing-comma --exit-zero-even-if-changed --py36-plus
	chorny .

.PHONY: lint
lint:
	flake8
	mypy .
	semgrep --config p/r2c-security-audit --severity ERROR --disable-version-check --error

PYTEST_WORKERS ?= $(shell getconf _NPROCESSORS_ONLN)
PYTEST_EXTRA ?=

.PHONY: test
test:
	pytest -s -n ${PYTEST_WORKERS} --timeout 300 ${PYTEST_EXTRA} medvedi
