# ActiveVisionBenchmark

A lightweight, **single-file** Python toolkit for building **visual stimuli** for active-vision experiments.

The primary focus is generating the oriented visual-flow stimulus described here:
https://laurentperrinet.github.io/sciblog/posts/2025-04-24-orienting-yourself-in-the-visual-flow.html

> **Paper:** see [`paper/ActiveVisionBenchmark.tex`](paper/ActiveVisionBenchmark.tex)

---

## Features

- 🧭 **Visual-flow stimulus** – `make_visual_flow_stimulus` builds oriented local-element patterns.
- 🖼️ **Base generators** – Gaussian noise and Gabor patches for controlled low-level stimuli.
- 🧪 **Reproducibility** – explicit random seeds for deterministic stimulus creation.
- 🔬 **Optional benchmarking layer** – existing task/model/metric tools are still available.

---

## Installation

No installation required — just copy [`ActiveVisionBenchmark.py`](ActiveVisionBenchmark.py)
into your project.

**Dependencies:**

| Package | Required | Notes |
|---------|----------|-------|
| `numpy` | ✅ | Core numerics |
| `scipy` | optional | Foveation, saliency, *d'* |
| `matplotlib` | optional | `plot_trial`, `plot_roc` |

```bash
pip install numpy
pip install numpy scipy matplotlib
```

---

## Quick Start

```python
from ActiveVisionBenchmark import make_visual_flow_stimulus

stimulus = make_visual_flow_stimulus(
    size=(256, 256),
    n_elements=128,
    focus_of_expansion=(0.0, 0.0),
    orientation_offset=3.14159 / 2,  # orthogonal to local flow direction
    seed=0,
)
```

---

## API Overview

| Symbol | Kind | Purpose |
|--------|------|---------|
| `make_gaussian_noise` | function | 2-D Gaussian noise array |
| `make_gabor_patch` | function | Gabor patch stimulus |
| `make_visual_flow_stimulus` | function | Oriented visual-flow stimulus image |
| `make_search_array` | function | Visual-search canvas |
| `VisualSearchTask` | class | Trial generator |
| `Trial` | dataclass | Single experimental trial |
| `ActiveVisionModel` | ABC | Custom model interface |
| `RandomSearchModel` | class | Uniform-random baseline |
| `SaliencySearchModel` | class | Contrast-saliency + IoR model |
| `foveate` | function | Eccentricity-dependent blurring |
| `Benchmark` | class | Run & compare models on trials |
| `BenchmarkResult` | dataclass | Metrics container |
| `plot_trial` | function | Scanpath overlay (matplotlib) |
| `plot_roc` | function | ROC-space plot (matplotlib) |

---

## Optional: Benchmarking Utilities

```python
from ActiveVisionBenchmark import ActiveVisionModel
import numpy as np

class CentreFixationModel(ActiveVisionModel):
    """Example model for the optional benchmark layer."""

    def reset(self, image):
        self._shape = image.shape[:2]

    def next_fixation(self, foveated_image, fixation_history):
        h, w = self._shape
        return h // 2, w // 2

    def make_decision(self):
        return True
```

---

## Paper

The LaTeX sources are in [`paper/`](paper).
The bibliography file is [`paper/ActiveVisionBenchmark.bib`](paper/ActiveVisionBenchmark.bib).

The PDF is compiled automatically by CI whenever `paper/` changes (see
[`.github/workflows/compile-tex.yml`](.github/workflows/compile-tex.yml))
and committed back as `paper/ActiveVisionBenchmark.pdf`.

To compile locally:

```bash
cd paper
latexmk -pdf -bibtex -interaction=nonstopmode ActiveVisionBenchmark.tex
```

---

## License

This project is distributed under the **GNU General Public License v3.0**.
See [`LICENSE`](LICENSE) for the full text.
