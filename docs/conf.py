# -*- coding: utf-8 -*-
"""Sphinx documentation configuration for ActiveVisionBenchmark.

Sources live in ``docs/``: Markdown pages and copies of the Jupyter notebooks of
``notebooks/`` (created by ``make notebooks``) are built as first-class pages, while
the single-file library ``ActiveVisionBenchmark.py`` is rendered from its docstrings
via autodoc. The built HTML is deployed to GitHub Pages by the workflow
``.github/workflows/docs.yml``, from the public repository only.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# -- Project information ------------------------------------------------------
project = "ActiveVisionBenchmark"
copyright = "2026, Laurent U Perrinet"
author = "Laurent U Perrinet"

version_raw = "unknown"
try:
    from ActiveVisionBenchmark import __version__ as version_raw   # single source of truth
except Exception:
    pass

# The Pages build passes `_RTD_VERSION` to avoid caching stale values across branches.
version = os.environ.get("_RTD_VERSION", version_raw)
release = version


# -- General Sphinx settings --------------------------------------------------
extensions = [
    "myst_parser",                  # Markdown pages
    "nbsphinx",                     # Jupyter notebook rendering + output embedding
    "sphinx.ext.autodoc",           # API page from docstrings
    "sphinx.ext.napoleon",          # Google-style docstrings
    "sphinx.ext.mathjax",           # equations of the docstrings and of the notebooks
    "sphinx.ext.intersphinx",
]

exclude_patterns = [
    "_build",
    ".venv/",
    "**/.ipynb_checkpoints",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# The dataclasses document their fields in an `Attributes:` section: rendering it as
# `:ivar:` fields avoids describing every field twice (once there, once by :undoc-members:).
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}


# -- HTML output ---------------------------------------------------------------
templates_path = []
html_static_path = []

html_theme = "furo"
html_title = "ActiveVisionBenchmark"

html_theme_options = {
    "source_repository": "https://github.com/laurentperrinet/ActiveVisionBenchmark/",
    "source_branch": "main",
    "source_directory": "docs/",
}


# -- nbsphinx rendering --------------------------------------------------------
# The long computations are cached locally (see ``avb.cached``) and the notebooks are
# executed there, so the stored outputs are rendered as they are: the CI never
# recomputes anything.
nbsphinx_execute = "never"
nbsphinx_allow_errors = False
