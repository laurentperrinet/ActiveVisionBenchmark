# Installation and quick start

## Installation

The library is a single file. There are three ways to get it.

**1. Copy the file.** Copy
[`ActiveVisionBenchmark.py`](https://github.com/laurentperrinet/ActiveVisionBenchmark/blob/main/ActiveVisionBenchmark.py)
next to your script or notebook. Only `numpy`, `torch` and `torchvision` are required:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy
```

**2. With pip, from the repository** (this also installs the dependencies):

```bash
pip install git+https://github.com/laurentperrinet/ActiveVisionBenchmark
# or, from a clone:
pip install -e .
```

**3. From PyPI**, once the package is published under the name `avb`:

```bash
pip install avb
```

Whatever the way, the module is imported as

```python
import ActiveVisionBenchmark as avb
```

Optional dependencies: `matplotlib` (figures), `imageio` and `imageio-ffmpeg` (videos),
`ipython` (embedded videos), `MotionClouds` (the texture generator of the Motion Clouds package; a built-in generator is
used if it is missing). `pip install "avb[full]"` installs them all. The step-by-step guide
is the notebook {doc}`notebooks/00_installation`.

On Apple silicon the computations use the Metal backend (`mps`) for the large images, and the
CPU for the small ones, where it is faster.

## Quick start

```python
import ActiveVisionBenchmark as avb

# A texture, projected on an infinite ground plane seen by a camera that moves on it.
task = avb.HorizontalPlaneTask(avb.MotionCloudStimulus(), size=128)
bench = avb.Benchmark(task)

# 200 reproducible trials, and the bundled models evaluated on them.
trials = bench.generate_dataset(n_trials=200, seed=0)
results = bench.evaluate(trials)
for name, res in results.items():
    print(res.summary())
```

A **passive** stream of images, with the ground-truth optic flow, in a tunnel:

```python
task = avb.CylinderTask(avb.GridStimulus(colour=True), size=128)
for frame in avb.PassiveBenchmark(task).stream(n_frames=60, seed=0):
    frame.image, frame.flow, frame.pose
```

An **active** model is a sub-class of {class}`~ActiveVisionBenchmark.ActiveVisionModel` that
returns an {class}`~ActiveVisionBenchmark.Action` from an
{class}`~ActiveVisionBenchmark.Observation`:

```python
class CentreGazeModel(avb.ActiveVisionModel):
    """Rotate the gaze towards the estimated focus of expansion."""

    def reset(self, seed=None):
        self.gaze_before = None

    def act(self, observation):
        foe = 0.0
        if observation.previous is not None:
            # The rotation of the gaze shifts the whole image: it is known, so it is removed.
            rotation = 0.0 if self.gaze_before is None else observation.gaze - self.gaze_before
            foe = avb.estimate_foe(observation.previous, observation.image, visual_angle=observation.visual_angle,
                                   rotation=rotation, yaw_range=0.2)
        self.gaze_before = observation.gaze
        return avb.Action(gaze_shift=0.4 * foe, heading_estimate=observation.gaze + foe)

task = avb.CylinderTask(avb.MotionCloudStimulus())
results = avb.ActiveBenchmark(task).evaluate(task.generate_dataset(10), {"mine": CentreGazeModel()})
print(results["mine"].summary())
```

## Videos

The streams of images are meant to be watched. Each video is a `.mp4` of a few hundred kilobytes that
every browser plays:

```python
tasks = {"ground plane": avb.HorizontalPlaneTask(avb.MotionCloudStimulus(), size=192),
         "tunnel": avb.CylinderTask(avb.MotionCloudStimulus(), size=192)}
# One panel per task; a dot is the true focus of expansion (the target) and a circle its estimate.
path = avb.passive_video(tasks, "navigation", duration=5.0, show_foe=True)
avb.show_video(path)        # in a notebook, the video is embedded in the output of the cell
```

* {func}`~ActiveVisionBenchmark.passive_video` plays passive streams side by side, for instance one
  panel per stimulus, and can draw the true and the estimated focus of expansion.
* {func}`~ActiveVisionBenchmark.landing_video` plays the same landing with several models, with the
  height of the camera at the bottom right.
* {func}`~ActiveVisionBenchmark.save_video` writes any list of images, and
  {func}`~ActiveVisionBenchmark.show_video` embeds a video in a notebook (so that it is also shown by
  this documentation, which never executes the notebooks).

Like the figures, a video is written to `figures/` only when `AVB_FIGURES=1` is set; otherwise it is
written to a temporary directory, which is enough to embed it. Writing videos needs `imageio` and
`imageio-ffmpeg`.

## Notebooks

The notebooks only import `ActiveVisionBenchmark`. Their long computations are cached in
`cache/`, and the figures are only written to `figures/` when the switch `AVB_FIGURES=1` is set,
that is when the paper is compiled. The notebooks are rendered in this
documentation, in this order: installation, stimuli and scenarios, navigation (infinite
plane and cylinder), perturbation, and landing.
