# ActiveVisionBenchmark (AVB)

Python library for neuroscience active vision benchmarking in `avbpy/`. Models gaze-driven visual search with two core actions: `WhereToLookFirstAction` and `WhereToLookNextAction` — powers both the codebase and the accompanying paper.

## Layout (the parts that matter)
- **`avbpy/benchmarking/`** — library source (`WhereToLook*Action`, core API).
- **`ActiveVisionBenchmark.py`** — primary entry script; also usable as a module.
- **`paper/`** — the accompanying paper (`.tex` + `.bib`). Derived artifacts (`*.pdf`) are gitignored; weekend CI re-compiles the PDF from source via `compile-tex.yml`.
- **`_dist_build_dist/`** — generated distribution artefacts (clean if stale).
- **`tests/data/`** — fixture data many benchmark suites depend on. When running benchmarks, this must be mounted into any isolation environment where scripts execute; failing to do so causes silent "missing file" errors that are confusing to debug.

## Commands (all verified)
```bash
# Single benchmark experiment with isolation env:
AVB_PYTHON=1 python -m avbpy run_benchmark --experiment 4

# Full CI-equivalent workflow: runs benchmarks + all tests in one go.
python _run_ci.py full

# Paper rebuild from source (with BibTeX fallback chain):
rm paper/ActiveVisionBenchmark.bbl-SAVE-ERROR 2>/dev/null; \
latexmk -pdf -pdflatex='pdflatex -interaction=nonstopmode' paper/ActiveVisionBenchmark.tex || { \
    bibtex paper/ActiveVisionBenchmark && latexmk -pdf paper/ActiveVisionBenchmark.tex; }

# Weekend CI build runs on master only. Source config at .github/workflows/compile-tex.yml
```

## Build and test quirks
- **dist-build**: package_name = `active-vision-benchmark`; layout declared in `build_configs/dist-build.conf.toml`. License collections referenced from `licenses/LIC_FILES_TXT/` and `LICENSES_SOURCE_PY/`.
- The library uses PEP 680 namespace-style installs (`package_style = "pep-680"`). Test scripts under `.avbpy_test_commands/*_run_tests_*_script.py` are gitignored because they share state with benchmarks.- **`paper/`**: `.tex` + `.bib` sources compiled into a paper PDF every weekend CI run; see `build/latexmk_paper.py`. Any library change worth documenting → update the paper too before opening a PR. When you touch one artifact, verify whether the other needs corresponding updates.

## Layout (high signal)

- **`avbpy/benchmarking/`** — `WhereToLookFirstAction`, `WhereToLookNextAction` (the core API that powers both Python + paper).
- **`avbpy/datasets/`** — dataset loading helpers; uses fixtures in this repo's `tests/data/`.
- **`_dist_build_dist/`** — generated build artefacts. Clean before rebuilding; config is at `build_configs/dist-build.conf.toml`.
- **`paper/PapersToSubmit/`** — paper build helpers (the actual LaTeX compile scripts). LaTeX build helpers live in `build/PapersToSubmit/` and `pandoc/compile_to_pdf_via_Python.py`.
- **`tests/data/`** — fixture data many benchmarks rely on; the CI test scripts under `.avbpy_test_commands/*_run_tests_*_script.py` are gitignored.

## Commands (verified)

```bash
# Single benchmark run with isolation env to keep dev state separate from benchmark envs:
AVB_PYTHON=1 python -m avbpy run_benchmark --experiment 4

# Full CI-equivalent workflow — benchmarks + all tests in one go
python _run_ci.py full

# Paper rebuild
rm paper/ActiveVisionBenchmark.bbl-SAVE-ERROR && \
    latexmk -pdf -pdflatex='pdflatex -interaction=nonstopmode' paper/ActiveVisionBenchmark.tex || {
        echo "BibTeX run required" && bibtex paper/ActiveVisionBenchmark && \
        latexmk -pdf paper/ActiveVisionBenchmark.tex
    }

# Paper tests (in CI) check that the build pipeline produces artefacts
python pandoc/compile_to_pdf_via_Python.py # in CI workflow; runs on master only via GitLab
```

## Tests & fixtures

- Required fixture data: `tests/data/` — used by many benchmark suites. When running benchmarks, this is mounted into an isolation container to keep the main Python env clean of test-data state changes.
- Test scripts are organized as CI-invoked subsets (`.avbpy_test_commands/*_run_tests_*_script.py`) because of how they share data with benchmarks — these are listed in CI configs under `tests/test_data/` and `.gitignore`.

## Build, dist & quirks

- **dist-build**: Uses the layout declared at `build_configs/dist-build.conf.toml` (package name: `active-vision-benchmark`; license collections referenced for packaging). Artifacts land in `_dist_build_dist/`.
  ```bash
  python -m avbpy dist build --build-dir=_dist_build_dist
  ```
- **Paper artefacts**: The `.bbl-SAVE-ERROR` file indicates that `bib` failed at a previous step. If you see it, run the paper commands above; they will retry with full rebuilds of the PDF.
- **Weekend CI LaTeX build** runs on master only and is configured as an isolated workflow — don't touch `.bbl`, `.log` or compiled outputs in PR branches until reviewed against a clean git clone of `master`.

## Gotchas specific to this agent-session pattern (active vision + data science)

- If you notice that test fixtures aren't available in current branch state, check `_run_ci.py full`; it will run them using the CI-configured fixture location, not wherever tests/data lives locally. Always mount the `tests/data/` directory into any isolation environment where benchmark scripts are invoked — otherwise data loading will fail silently or with confusing errors about missing files.
- The library is designed to keep Python state clean from testing state, so **don't** run test suites in your main Python session unless that's on purpose (it won't break things but it could affect how a fresh agent session looks at what you've done).

## References

- Source config: `build_configs/dist-build.conf.toml`
- CI workflow source-of-truth for LaTeX pipeline: `.github/workflows/compile-tex.yml`
