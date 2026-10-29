# Makefile for the ActiveVisionBenchmark documentation.
#
# The site is built into docs/_build/html (git-ignored); only the sources
# (conf.py, *.md, requirements.txt, this Makefile) are versioned. The GitHub
# Actions workflow .github/workflows/docs.yml runs `make -C docs html` and
# deploys docs/_build/html to GitHub Pages on every push to main of the public
# repository.
#
#   pip install -r docs/requirements.txt
#   make -C docs html        # copy the notebooks + build the site
#   open docs/_build/html/index.html

SPHINXBUILD ?= sphinx-build
SPHINXOPTS  ?=
BUILDDIR     = _build

NOTEBOOKS := $(notdir $(wildcard ../notebooks/*.ipynb))

.PHONY: help notebooks html clean all

help:
	@echo "Available targets:"
	@echo "  notebooks  (re)copy ../notebooks/*.ipynb into notebooks/"
	@echo "  html       build the HTML site in $(BUILDDIR)/html"
	@echo "  clean      remove generated artefacts (keeps conf.py, Makefile, *.md)"
	@echo "  all        notebooks + html (default)"

# Copies rather than symlinks: Sphinx stores the outputs of a notebook next to its
# page, which does not work well through a link that leaves the source directory.
notebooks:
	@mkdir -p notebooks
	@rm -f notebooks/*.ipynb
	@for nb in $(NOTEBOOKS); do \
		cp ../notebooks/$$nb notebooks/$$nb ; \
	done

html: notebooks
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -d $(BUILDDIR)/doctrees . $(BUILDDIR)/html
	@echo "Site ready in $(BUILDDIR)/html"

clean:
	rm -rf $(BUILDDIR) notebooks

all: html
