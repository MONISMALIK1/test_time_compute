.PHONY: help test install clean solve bench

help:		## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

test:		## Run the offline test suite (no API key needed)
	cd .. && python -m unittest discover -s test_time_compute/tests -t . -v

install:	## Editable install of the package
	pip install -e .

solve:		## Solve a problem: make solve ARGS='"A train goes 60mph for 3h. How far?" --show-candidates'
	python -m test_time_compute $(ARGS)

bench:		## Benchmark TTC vs the n=1 baseline on the bundled math set
	python -m test_time_compute --bench

clean:		## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info build dist .eggs
