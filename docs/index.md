# ActiveVisionBenchmark

A lightweight, **single-file** Python library to benchmark *active vision*: it builds
controlled visual-flow stimuli (a tunnel, an infinite ground plane) and probes them with a
camera that is either **passive** (it generates streams of images) or **active** (it is
controlled by a model that closes the sensorimotor loop).

```{eval-rst}
.. image:: https://img.shields.io/badge/License-GPL--3.0-green.svg
   :target: https://github.com/laurentperrinet/ActiveVisionBenchmark/blob/main/LICENSE
```

## The three layers

| Layer | Objects | Role |
|-------|---------|------|
| **Stimulus** | {class}`~ActiveVisionBenchmark.GridStimulus`, {class}`~ActiveVisionBenchmark.MotionCloudStimulus`, {class}`~ActiveVisionBenchmark.GaussianNoiseStimulus`, {class}`~ActiveVisionBenchmark.BinaryNoiseStimulus`, {class}`~ActiveVisionBenchmark.DotNoiseStimulus`, {class}`~ActiveVisionBenchmark.ImageStimulus`, {class}`~ActiveVisionBenchmark.ArrayStimulus` | create the texture image |
| **Task** | {class}`~ActiveVisionBenchmark.CylinderTask`, {class}`~ActiveVisionBenchmark.HorizontalPlaneTask`, {class}`~ActiveVisionBenchmark.LandingTask`, {class}`~ActiveVisionBenchmark.Perturbation` | project the texture onto a 3D geometry seen by a pinhole camera, move the camera, give the ground-truth optic flow |
| **Benchmark** | {class}`~ActiveVisionBenchmark.Benchmark`, {class}`~ActiveVisionBenchmark.PassiveBenchmark`, {class}`~ActiveVisionBenchmark.ActiveBenchmark` | evaluate models on a dataset of reproducible trials |

## The three benchmarks

1. **Navigation** (plane and cylinder; see {doc}`notebooks/02_navigation_plane_cylinder`). The observer estimates the direction of its
   egomotion (plane), or keeps its gaze on the axis of a tunnel (cylinder), while the target
   (the focus of expansion, FoE) diffuses slowly.
2. **Perturbation** (see {doc}`notebooks/03_perturbation`). Abrupt exogenous jumps of the FoE (a hazard rate and a Gaussian
   amplitude) are added.
3. **Landing** (see {doc}`notebooks/04_landing`). The camera descends onto the ground plane and has to touch it as fast and
   as softly as possible.

## Where to go next

Start with the {doc}`quickstart`. The notebooks are numbered and build on one another: they
are rendered here with the outputs they were executed with. The equations of the model and
the table of its variables are at the top of the {ref}`API reference <api-reference>`.

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
notebooks/00_installation
notebooks/01_stimuli_and_scenarios
notebooks/02_navigation_plane_cylinder
notebooks/03_perturbation
notebooks/04_landing
api
```
