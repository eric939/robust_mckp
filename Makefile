PYTHON ?= python3
TECTONIC ?= tectonic
V4_RELEASE_RESULTS ?= results/v4_publication_20260721_certified_final
V4_RUN_RESULTS ?= results/v4_reproduction
V4_RUN_PAPER ?= tmp/v4_reproduction_paper
V4_CALIBRATION ?= $(V4_RELEASE_RESULTS)/uci_calibration
V4_EXTERNAL_ARCHIVE ?= data_cache/RobustKnapsack.zip

.PHONY: install-dev test check clean-preview \
	v4-verify v4-evidence v4-exact-audit v4-reproduce v4-paper v4-package \
	v4-anonymous-package clean

install-dev:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[experiments,validation,dev]"

test:
	$(PYTHON) -m pytest -q

check: v4-verify

v4-verify: test
	MPLCONFIGDIR="$$HOME/.cache/matplotlib" $(PYTHON) scripts/verify_v4_release.py \
		--results $(V4_RELEASE_RESULTS) \
		--paper paper_versions/v4

v4-evidence:
	MPLCONFIGDIR="$$HOME/.cache/matplotlib" $(PYTHON) research/generate_v4_publication_artifacts.py \
		--results $(V4_RELEASE_RESULTS) \
		--paper paper_versions/v4

v4-exact-audit:
	env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
		$(PYTHON) research/exact_integration_campaign.py \
		--output-dir $(V4_RELEASE_RESULTS)/exact_integration \
		--sizes 30 --seeds 0,1,2 --repetitions 2 --time-limit 5

v4-reproduce: test
	env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	$(PYTHON) scripts/run_v4_publication_campaign.py \
		--output-dir $(V4_RUN_RESULTS) \
		--calibration-dir $(V4_CALIBRATION) \
		--external-archive $(V4_EXTERNAL_ARCHIVE)
	env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
		$(PYTHON) research/exact_integration_campaign.py \
		--output-dir $(V4_RUN_RESULTS)/exact_integration \
		--sizes 30 --seeds 0,1,2 --repetitions 2 --time-limit 5
	MPLCONFIGDIR="$$HOME/.cache/matplotlib" $(PYTHON) research/generate_v4_publication_artifacts.py \
		--results $(V4_RUN_RESULTS) \
		--paper $(V4_RUN_PAPER)

v4-paper:
	cd paper_versions/v4 && $(TECTONIC) main_v4.tex
	cd paper_versions/v4 && $(TECTONIC) main_v4_opre.tex
	cd paper_versions/v4 && $(TECTONIC) main_v4_opre_blind.tex
	cd paper_versions/v4 && $(TECTONIC) main_v4_ec.tex
	cd paper_versions/v4 && $(TECTONIC) main_v4_ec_blind.tex
	cd paper_versions/v4 && $(TECTONIC) executive_summary_opre.tex

v4-package: v4-paper
	mkdir -p output/pdf
	cp paper_versions/v4/main_v4.pdf output/pdf/robust_mckp_v4_full.pdf
	cp paper_versions/v4/main_v4_opre.pdf output/pdf/robust_mckp_v4_opre.pdf
	cp paper_versions/v4/main_v4_opre_blind.pdf output/pdf/robust_mckp_v4_opre_blind.pdf
	cp paper_versions/v4/main_v4_ec.pdf output/pdf/robust_mckp_v4_electronic_companion.pdf
	cp paper_versions/v4/main_v4_ec_blind.pdf output/pdf/robust_mckp_v4_electronic_companion_blind.pdf
	cp paper_versions/v4/executive_summary_opre.pdf output/pdf/robust_mckp_v4_executive_summary.pdf

v4-anonymous-package: v4-verify v4-paper
	$(PYTHON) scripts/build_v4_anonymous_supplement.py \
		--results $(V4_RELEASE_RESULTS) \
		--paper paper_versions/v4 \
		--output output/anonymous/robust_mckp_v4_anonymous_supplement.zip

clean-preview:
	$(PYTHON) scripts/clean_workspace.py

clean:
	$(PYTHON) scripts/clean_workspace.py --apply
