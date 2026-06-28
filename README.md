# ActiveVisionBenchmark

ActiveVisionBenchmark is a lightweight Python toolkit to build and benchmark **active vision** tasks.
It focuses on controlled visual stimulation for studying perception in closed sensorimotor loops,
bridging classical simplified stimuli and more ecological visual scenarios.

This repository follows the motivation outlined in the paper introduction: preserving experimental
control while extending benchmarks toward richer, action-dependent visual contexts.

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
pip install numpy
pip install numpy scipy matplotlib
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

## Paper

The LaTeX sources are in [`paper/`](paper).
The bibliography file is [`paper/AVB.bib`](paper/AVB.bib).

To compile:

```bash
cd paper
pdflatex ActiveVisionBenchmark.tex
bibtex ActiveVisionBenchmark
pdflatex ActiveVisionBenchmark.tex
pdflatex ActiveVisionBenchmark.tex
```

---

## License

This project is distributed under the **GNU General Public License v3.0**.
See [`LICENSE`](LICENSE) for the full text.
