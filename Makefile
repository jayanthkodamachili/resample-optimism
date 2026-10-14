.PHONY: install lint test experiments figures receipts verify clean

install:
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest --cov=resample_optimism --cov-report=term-missing --cov-fail-under=100

experiments:
	python experiments/e1_headline.py
	python experiments/e2_by_resampler.py
	python experiments/e3_by_estimator.py
	python experiments/e4_was_it_worth_it.py

figures:
	python tools/render_diagram.py
	python tools/capture_cli.py

# Re-measures everything, then refuses to pass if a document quotes a figure the code no
# longer produces. Use --reuse locally to skip the twelve minute experiment re-run.
receipts:
	python tools/collect_metrics.py
	python tools/check_numbers.py

receipts-fast:
	python tools/collect_metrics.py --reuse
	python tools/check_numbers.py

verify: lint test receipts

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.json junit.xml .coverage
