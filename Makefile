.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests retrain_baseline recreate_dev_api

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

test:
	python -m pytest $(TEST_FILE)

integration_tests:
	python -m pytest tests/integration_tests 

test_watch:
	python -m ptw --snapshot-update --now . -- -vv tests/unit_tests

test_profile:
	python -m pytest -vv tests/unit_tests/ --profile-svg

extended_tests:
	python -m pytest --only-extended $(TEST_FILE)


######################
# INFERENCE-MODEL SWITCH
######################

# The dev API container. Its interpreter carries the PINNED scikit-learn, so the
# baseline pickles it fits are the ones the deployment can load, and its network
# resolves the store URI (host.docker.internal does not resolve on the host).
DEV_API_CONTAINER ?= anubis-dev-langgraph-api-dev-1

# Switch the inference model and rebuild every baseline artifact in one command
# (wraps scripts/switch_inference_model.sh, which can also be run directly):
#   make retrain_baseline ARGS='--model gpt-5.6-luna --model-provider OPEN_AI \
#       --model-prompt-cost 0.0000002 --model-completion-cost 0.0000012'
# Then `make recreate_dev_api`: env_file is read when a container is CREATED, so
# the rewritten MODEL / costs / BASELINE_RESPONSE_THRESHOLD only reach the API
# after a recreate (a plain restart keeps the old values).
retrain_baseline:
	API_CONTAINER=$(DEV_API_CONTAINER) ./scripts/switch_inference_model.sh $(ARGS)

recreate_dev_api:
	docker compose --env-file .env.dev up -d --force-recreate --no-deps langgraph-api-dev


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=src/
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	python -m ruff check .
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || python -m mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && python -m mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	ruff format $(PYTHON_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'retrain_baseline ARGS=...    - switch MODEL + costs and rebuild the style baseline (in the dev container)'
	@echo 'recreate_dev_api             - recreate the dev API container so it adopts the rewritten env'

