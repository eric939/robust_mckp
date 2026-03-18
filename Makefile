PYTHON ?= python3

.PHONY: install-dev test paper-fast paper-nested paper-case paper-all paper-validate clean-paper

install-dev:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[experiments,validation,dev]"

test:
	$(PYTHON) -m pytest -q

paper-fast:
	$(PYTHON) experiments_nested/exp1_integrality_gap.py --fast --output-dir paper/exp1 --results-dir paper/results_nested
	$(PYTHON) experiments_nested/exp2_scalability.py --fast --output-dir paper/exp2 --results-dir paper/results_nested
	$(PYTHON) experiments_nested/exp3_risk_frontier.py --fast --output-dir paper/exp3 --results-dir paper/results_nested
	$(PYTHON) experiments_nested/exp4_summary_table.py --fast --results-dir paper/results_nested --tables-dir paper/tables_nested
	$(PYTHON) experiments_case_retail/case_retail_pricing.py --fast --scenarios 2000 --output-dir "paper/retail pricing" --results-dir paper/results_case

paper-nested:
	$(PYTHON) experiments_nested/exp1_integrality_gap.py --output-dir paper/exp1 --results-dir paper/results_nested
	$(PYTHON) experiments_nested/exp2_scalability.py --output-dir paper/exp2 --results-dir paper/results_nested
	$(PYTHON) experiments_nested/exp3_risk_frontier.py --output-dir paper/exp3 --results-dir paper/results_nested
	$(PYTHON) experiments_nested/exp4_summary_table.py --results-dir paper/results_nested --tables-dir paper/tables_nested

paper-case:
	$(PYTHON) experiments_case_retail/case_retail_pricing.py --output-dir "paper/retail pricing" --results-dir paper/results_case

paper-all: paper-nested paper-case

paper-validate:
	$(PYTHON) experiments_nested/exp1_integrality_gap.py --enable-milp --global-milp --output-dir paper/exp1 --results-dir paper/results_nested
	$(PYTHON) experiments_nested/exp2_scalability.py --validate-lp --output-dir paper/exp2 --results-dir paper/results_nested
	$(PYTHON) experiments_nested/exp4_summary_table.py --enable-milp --results-dir paper/results_nested --tables-dir paper/tables_nested

clean-paper:
	rm -rf paper/exp1 paper/exp2 paper/exp3 "paper/retail pricing" paper/results_case paper/results_nested paper/tables_nested
