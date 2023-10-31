style:
	black src
	ruff --fix src

check:
	black --check src
	ruff src
	pylint -j 8 --fail-under=8 src

test:
	rm -r build
	rm -r src/slow_fast.egg-info
	pip install -e .
	pytest tests