# ActiveVisionBenchmark

A lightweight, **single-file** Python benchmarking toolkit for **active vision** systems.

Active vision refers to perception systems that actively control where they direct
their gaze or attention to optimally gather information from a visual scene.

> **Paper:** see [`paper/ActiveVisionBenchmark.tex`](paper/ActiveVisionBenchmark.tex)

---

## Features

- 🖼️ **Stimulus generators** – Gaussian noise, Gabor patches, visual-search arrays.
- 🔬 **Task abstraction** – `VisualSearchTask` produces reproducible `Trial` streams.
- 🤖 **Model interface** – plug in any algorithm via `ActiveVisionModel`.
- 👁️ **Foveation** – eccentricity-dependent resolution drop simulated with Gaussian blur.
- 📊 **Metrics** – accuracy, hit/false-alarm rates, *d'*, mean fixations, search efficiency.
- 📈 **Visualisation** – scanpath plots and ROC curves (requires `matplotlib`).

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
pip install numpy           # minimum
pip install numpy scipy matplotlib  # full feature set
```

---

## Quick Start

```python
from ActiveVisionBenchmark import (
    Benchmark, RandomSearchModel, SaliencySearchModel, VisualSearchTask,
)

# 1. Generate trials
task   = VisualSearchTask(image_size=(256, 256), n_distractors=10, seed=0)
trials = task.generate_dataset(n_trials=100)

# 2. Define models
models = {
    "Random":   RandomSearchModel(seed=1),
    "Saliency": SaliencySearchModel(window_size=16, inhibition_radius=20),
}

# 3. Run benchmark
bench   = Benchmark(max_fixations=20)
results = bench.compare(models, trials)

# 4. Print summaries
for name, res in results.items():
    print(res.summary())
```

---

## API Overview

| Symbol | Kind | Purpose |
|--------|------|---------|
| `make_gaussian_noise` | function | 2-D Gaussian noise array |
| `make_gabor_patch` | function | Gabor patch stimulus |
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

## Implementing a Custom Model

```python
from ActiveVisionBenchmark import ActiveVisionModel
import numpy as np

class CentreFixationModel(ActiveVisionModel):
    """Always fixates the image centre, then guesses present."""

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

The accompanying LaTeX paper is in [`paper/ActiveVisionBenchmark.tex`](paper/ActiveVisionBenchmark.tex).
To compile:

```bash
cd paper
pdflatex ActiveVisionBenchmark.tex
pdflatex ActiveVisionBenchmark.tex   # second pass for cross-references
```

---

## License

[MIT](LICENSE)