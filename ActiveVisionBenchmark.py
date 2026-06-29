"""
ActiveVisionBenchmark
=====================

A lightweight, single-file toolkit centered on creating controlled visual
stimuli, including oriented visual-flow stimuli for active-vision experiments.

The core stimulus target is the visual-flow orientation setup described in:
https://laurentperrinet.github.io/sciblog/posts/2025-04-24-orienting-yourself-in-the-visual-flow.html

This library provides:

* Low-level stimulus generators (noise fields, Gabor patches).
* A visual-flow stimulus constructor with oriented local elements.
* Optional benchmark/task utilities for model comparison workflows.

Dependencies: numpy (required), scipy and matplotlib (optional).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "0.1.0"
__author__ = "Laurent Perrinet"
__license__ = "GPL-3.0-or-later"

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` from *seed*."""
    return np.random.default_rng(seed)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr.mean(axis=-1)
    raise ValueError(f"Expected 2-D or 3-D array, got shape {arr.shape}")


# ---------------------------------------------------------------------------
# Stimulus generators
# ---------------------------------------------------------------------------


def make_gaussian_noise(
    size: Tuple[int, int] = (128, 128),
    mean: float = 0.5,
    std: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return a 2-D array of Gaussian noise in [0, 1].

    Parameters
    ----------
    size:
        ``(height, width)`` of the output array.
    mean:
        Mean luminance (0–1).
    std:
        Standard deviation.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape *size* clipped to [0, 1].
    """
    rng = _rng(seed)
    return np.clip(rng.normal(mean, std, size=size), 0.0, 1.0)


def make_gabor_patch(
    size: Tuple[int, int] = (32, 32),
    frequency: float = 0.1,
    orientation: float = 0.0,
    phase: float = 0.0,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """Return a Gabor patch (sinusoid × Gaussian envelope) in [0, 1].

    Parameters
    ----------
    size:
        ``(height, width)`` in pixels.
    frequency:
        Spatial frequency in cycles per pixel.
    orientation:
        Orientation in radians (0 = horizontal grating).
    phase:
        Phase offset in radians.
    sigma:
        Standard deviation of the Gaussian envelope in pixels.
        Defaults to ``min(size) / 4``.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape *size*.
    """
    h, w = size
    if sigma is None:
        sigma = min(h, w) / 4.0
    y = np.linspace(-h / 2, h / 2, h)
    x = np.linspace(-w / 2, w / 2, w)
    xx, yy = np.meshgrid(x, y)
    xr = xx * np.cos(orientation) + yy * np.sin(orientation)
    envelope = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * frequency * xr + phase)
    patch = envelope * sinusoid
    # Normalise to [0, 1]
    patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-12)
    return patch


def make_search_array(
    image_size: Tuple[int, int] = (256, 256),
    target: Optional[np.ndarray] = None,
    n_distractors: int = 10,
    distractor_fn: Optional[Callable[[], np.ndarray]] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """Compose a visual-search array with one optional target among distractors.

    Parameters
    ----------
    image_size:
        ``(height, width)`` of the canvas.
    target:
        2-D patch to embed as the target.  If *None*, no target is placed
        (target-absent trial).
    n_distractors:
        Number of distractor patches.
    distractor_fn:
        Callable that returns a 2-D patch for each distractor.  Defaults to
        random Gabor patches with a horizontal orientation.
    seed:
        Random seed.

    Returns
    -------
    image : numpy.ndarray
        Float64 image array of shape *image_size*.
    target_location : tuple[int, int] or None
        ``(row, col)`` of the target patch centre, or *None* for absent trials.
    """
    rng = _rng(seed)
    canvas = make_gaussian_noise(image_size, mean=0.5, std=0.05, seed=int(rng.integers(1 << 31)))

    patch_size = (32, 32) if target is None else target.shape[:2]

    if distractor_fn is None:

        def distractor_fn() -> np.ndarray:  # type: ignore[misc]
            return make_gabor_patch(patch_size, frequency=0.15, orientation=0.0)

    h, w = image_size
    ph, pw = patch_size

    def _random_position() -> Tuple[int, int]:
        row = int(rng.integers(ph // 2, h - ph // 2))
        col = int(rng.integers(pw // 2, w - pw // 2))
        return row, col

    def _paste(canvas: np.ndarray, patch: np.ndarray, centre: Tuple[int, int]) -> None:
        r, c = centre
        r0, r1 = r - ph // 2, r - ph // 2 + ph
        c0, c1 = c - pw // 2, c - pw // 2 + pw
        r0, r1 = max(r0, 0), min(r1, h)
        c0, c1 = max(c0, 0), min(c1, w)
        pr0, pr1 = r0 - (r - ph // 2), r1 - (r - ph // 2)
        pc0, pc1 = c0 - (c - pw // 2), c1 - (c - pw // 2)
        canvas[r0:r1, c0:c1] = patch[pr0:pr1, pc0:pc1]

    for _ in range(n_distractors):
        _paste(canvas, distractor_fn(), _random_position())

    target_location: Optional[Tuple[int, int]] = None
    if target is not None:
        target_location = _random_position()
        _paste(canvas, target, target_location)

    return canvas, target_location


def make_visual_flow_stimulus(
    size: Tuple[int, int] = (256, 256),
    n_elements: int = 128,
    focus_of_expansion: Tuple[float, float] = (0.0, 0.0),
    orientation_offset: float = np.pi / 2,
    element_length_sigma: float = 4.0,
    element_width_sigma: float = 1.2,
    contrast: float = 0.35,
    background_mean: float = 0.5,
    background_std: float = 0.03,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Create an oriented visual-flow stimulus image in [0, 1].

    The generated pattern follows a radial optic-flow field. Each local element
    is an anisotropic Gaussian "stroke" whose orientation is tied to the local
    flow direction plus ``orientation_offset``. By default, this produces
    orientations orthogonal to flow direction.

    Parameters
    ----------
    size:
        ``(height, width)`` of the stimulus image.
    n_elements:
        Number of local oriented elements drawn onto the image.
    focus_of_expansion:
        ``(row, col)`` offset of the flow origin, in pixels, relative to the
        image center.
    orientation_offset:
        Angular offset (radians) added to local flow direction.
    element_length_sigma:
        Standard deviation of the element's long axis (pixels).
    element_width_sigma:
        Standard deviation of the element's short axis (pixels).
    contrast:
        Amplitude scaling of each element contribution.
    background_mean:
        Mean luminance of the background.
    background_std:
        Standard deviation of the Gaussian-noise background.
    seed:
        Random seed for reproducibility.
    """
    rng = _rng(seed)
    h, w = size
    canvas = make_gaussian_noise(
        size=size,
        mean=background_mean,
        std=background_std,
        seed=int(rng.integers(1 << 31)),
    )

    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    centre_row = h / 2 + float(focus_of_expansion[0])
    centre_col = w / 2 + float(focus_of_expansion[1])

    min_margin = int(max(4 * element_length_sigma, 2))
    if h <= 2 * min_margin or w <= 2 * min_margin:
        raise ValueError("size is too small for requested element_length_sigma")

    for _ in range(n_elements):
        row = float(rng.integers(min_margin, h - min_margin))
        col = float(rng.integers(min_margin, w - min_margin))

        flow_angle = float(np.arctan2(row - centre_row, col - centre_col))
        theta = flow_angle + orientation_offset
        ct, st = np.cos(theta), np.sin(theta)

        x_rel = xx - col
        y_rel = yy - row
        x_rot = x_rel * ct + y_rel * st
        y_rot = -x_rel * st + y_rel * ct

        element = np.exp(
            -0.5
            * (
                (x_rot / max(element_length_sigma, 1e-6)) ** 2
                + (y_rot / max(element_width_sigma, 1e-6)) ** 2
            )
        )
        polarity = 1.0 if rng.random() > 0.5 else -1.0
        canvas += polarity * contrast * element

    return np.clip(canvas, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Task abstraction
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """A single experimental trial.

    Attributes
    ----------
    image:
        2-D float array representing the stimulus.
    target_present:
        Whether the target is present in the image.
    target_location:
        ``(row, col)`` of the target centre, or *None* when absent.
    metadata:
        Arbitrary extra information about the trial.
    """

    image: np.ndarray
    target_present: bool
    target_location: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualSearchTask:
    """Generate trials for a conjunction visual-search experiment.

    Parameters
    ----------
    image_size:
        ``(height, width)`` in pixels.
    n_distractors:
        Number of distractors per trial.
    target_present_prob:
        Probability that the target is present in a trial.
    seed:
        Master seed for reproducibility.

    Examples
    --------
    >>> task = VisualSearchTask(image_size=(64, 64), n_distractors=5, seed=0)
    >>> trial = task.generate_trial()
    >>> trial.image.shape
    (64, 64)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        n_distractors: int = 10,
        target_present_prob: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        self.image_size = image_size
        self.n_distractors = n_distractors
        self.target_present_prob = target_present_prob
        self._rng = _rng(seed)

        # Build a canonical target (tilted Gabor)
        self._target = make_gabor_patch(
            size=(32, 32), frequency=0.15, orientation=np.pi / 4
        )
        # Distractors: horizontal Gabor
        self._distractor = make_gabor_patch(
            size=(32, 32), frequency=0.15, orientation=0.0
        )

    def generate_trial(self) -> Trial:
        """Generate and return a single :class:`Trial`."""
        target_present = bool(self._rng.random() < self.target_present_prob)
        seed = int(self._rng.integers(1 << 31))
        image, location = make_search_array(
            image_size=self.image_size,
            target=self._target if target_present else None,
            n_distractors=self.n_distractors,
            distractor_fn=lambda: self._distractor.copy(),
            seed=seed,
        )
        return Trial(image=image, target_present=target_present, target_location=location)

    def generate_dataset(self, n_trials: int = 100) -> List[Trial]:
        """Generate a list of *n_trials* :class:`Trial` objects."""
        return [self.generate_trial() for _ in range(n_trials)]


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------


class ActiveVisionModel:
    """Abstract base class for active vision models.

    Sub-class this to implement a custom active-vision or visual-search
    algorithm.  Concrete models must override :meth:`reset` and
    :meth:`next_fixation`.

    The model interacts with a :class:`Benchmark` via the following protocol:

    1. :meth:`reset` is called once per trial with the full image.
    2. :meth:`next_fixation` is called repeatedly; the model receives a
       *foveated* view and returns ``(row, col)`` of the next fixation.
    3. :meth:`make_decision` is called at the end of the trial to obtain
       the model's target-present / absent decision.
    """

    def reset(self, image: np.ndarray) -> None:
        """Initialise model state for a new trial.

        Parameters
        ----------
        image:
            The full scene (2-D float array).
        """
        raise NotImplementedError

    def next_fixation(
        self, foveated_image: np.ndarray, fixation_history: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Propose the next fixation location.

        Parameters
        ----------
        foveated_image:
            The image processed through the simulated fovea (same shape as the
            full image, blurred outside the current fixation region).
        fixation_history:
            Ordered list of ``(row, col)`` fixations so far (may be empty on
            the first call).

        Returns
        -------
        tuple[int, int]
            ``(row, col)`` of the proposed next fixation.
        """
        raise NotImplementedError

    def make_decision(self) -> bool:
        """Return the target-present decision (``True`` = target present)."""
        raise NotImplementedError


class RandomSearchModel(ActiveVisionModel):
    """Baseline model that fixates uniformly at random.

    Parameters
    ----------
    seed:
        Random seed.

    Examples
    --------
    >>> model = RandomSearchModel(seed=0)
    >>> task = VisualSearchTask(image_size=(64, 64), seed=1)
    >>> trial = task.generate_trial()
    >>> model.reset(trial.image)
    >>> loc = model.next_fixation(trial.image, [])
    >>> len(loc)
    2
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = _rng(seed)
        self._image: Optional[np.ndarray] = None
        self._detections: List[bool] = []

    def reset(self, image: np.ndarray) -> None:
        self._image = image
        self._detections = []

    def next_fixation(
        self, foveated_image: np.ndarray, fixation_history: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        h, w = foveated_image.shape[:2]
        row = int(self._rng.integers(0, h))
        col = int(self._rng.integers(0, w))
        return row, col

    def make_decision(self) -> bool:
        return bool(self._rng.random() > 0.5)


class SaliencySearchModel(ActiveVisionModel):
    """Model that fixates the most salient (highest-contrast) region.

    Saliency is computed as the local RMS contrast in a sliding window.

    Parameters
    ----------
    window_size:
        Edge length (pixels) of the local contrast window.
    inhibition_radius:
        Radius (pixels) around previously visited fixations that is
        suppressed from the saliency map (inhibition of return).
    """

    def __init__(self, window_size: int = 16, inhibition_radius: int = 20) -> None:
        self.window_size = window_size
        self.inhibition_radius = inhibition_radius
        self._saliency: Optional[np.ndarray] = None

    def reset(self, image: np.ndarray) -> None:
        self._saliency = self._compute_saliency(_ensure_2d(image).copy())

    def _compute_saliency(self, image: np.ndarray) -> np.ndarray:
        """Local RMS contrast map."""
        try:
            from scipy.ndimage import uniform_filter  # type: ignore[import]
        except ImportError:  # pragma: no cover
            warnings.warn(
                "scipy is not installed; falling back to uniform saliency.",
                stacklevel=3,
            )
            return np.ones_like(image)
        w = self.window_size
        mean = uniform_filter(image, size=w)
        mean_sq = uniform_filter(image**2, size=w)
        variance = np.maximum(mean_sq - mean**2, 0.0)
        return np.sqrt(variance)

    def next_fixation(
        self, foveated_image: np.ndarray, fixation_history: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        assert self._saliency is not None, "Call reset() before next_fixation()."
        sal = self._saliency.copy()
        h, w = sal.shape

        # Inhibition of return
        r = self.inhibition_radius
        for row, col in fixation_history:
            r0, r1 = max(0, row - r), min(h, row + r + 1)
            c0, c1 = max(0, col - r), min(w, col + r + 1)
            sal[r0:r1, c0:c1] = 0.0

        idx = int(np.argmax(sal))
        return int(idx // w), int(idx % w)

    def make_decision(self) -> bool:
        # Simple heuristic: target present if max saliency exceeds threshold
        if self._saliency is None:
            return False
        return bool(self._saliency.max() > 0.15)


# ---------------------------------------------------------------------------
# Foveation
# ---------------------------------------------------------------------------


def foveate(
    image: np.ndarray,
    fixation: Tuple[int, int],
    fovea_radius: int = 16,
    periphery_sigma: float = 4.0,
) -> np.ndarray:
    """Return a foveated copy of *image* centred on *fixation*.

    The region within *fovea_radius* is kept at full resolution.  Outside the
    fovea, a Gaussian blur is applied with standard deviation proportional to
    eccentricity.

    Parameters
    ----------
    image:
        2-D float array.
    fixation:
        ``(row, col)`` of the current fixation point.
    fovea_radius:
        Radius (pixels) of the high-resolution foveal region.
    periphery_sigma:
        Sigma (pixels) of the Gaussian blur applied to the periphery.

    Returns
    -------
    numpy.ndarray
        Foveated image with the same shape as *image*.
    """
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore[import]
    except ImportError:
        warnings.warn("scipy not installed; foveation skipped.", stacklevel=2)
        return image.copy()

    image = _ensure_2d(image)
    from scipy.ndimage import gaussian_filter  # noqa: F811

    blurred = gaussian_filter(image, sigma=periphery_sigma)
    h, w = image.shape
    row, col = fixation

    # Build a mask: 1 inside fovea, 0 outside
    y_idx = np.arange(h)[:, None]
    x_idx = np.arange(w)[None, :]
    dist = np.sqrt((y_idx - row) ** 2 + (x_idx - col) ** 2)
    mask = (dist <= fovea_radius).astype(float)

    return mask * image + (1 - mask) * blurred


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Result of running one model on one trial.

    Attributes
    ----------
    trial:
        The original :class:`Trial`.
    fixations:
        Ordered list of ``(row, col)`` fixation locations.
    decision:
        Model's target-present decision.
    correct:
        Whether the decision matches the ground truth.
    elapsed_s:
        Wall-clock time (seconds) taken for this trial.
    """

    trial: Trial
    fixations: List[Tuple[int, int]]
    decision: bool
    correct: bool
    elapsed_s: float


@dataclass
class BenchmarkResult:
    """Aggregated results from a full benchmark run.

    Attributes
    ----------
    trial_results:
        Per-trial :class:`TrialResult` objects.
    model_name:
        Name of the evaluated model.
    """

    trial_results: List[TrialResult]
    model_name: str = "unknown"

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    @property
    def accuracy(self) -> float:
        """Fraction of correctly decided trials."""
        if not self.trial_results:
            return float("nan")
        return sum(r.correct for r in self.trial_results) / len(self.trial_results)

    @property
    def mean_fixations(self) -> float:
        """Mean number of fixations per trial."""
        if not self.trial_results:
            return float("nan")
        return float(np.mean([len(r.fixations) for r in self.trial_results]))

    @property
    def mean_elapsed_s(self) -> float:
        """Mean wall-clock time per trial."""
        if not self.trial_results:
            return float("nan")
        return float(np.mean([r.elapsed_s for r in self.trial_results]))

    def hit_rate(self) -> float:
        """True-positive rate (hits / target-present trials)."""
        present = [r for r in self.trial_results if r.trial.target_present]
        if not present:
            return float("nan")
        return sum(r.decision for r in present) / len(present)

    def false_alarm_rate(self) -> float:
        """False-positive rate (false alarms / target-absent trials)."""
        absent = [r for r in self.trial_results if not r.trial.target_present]
        if not absent:
            return float("nan")
        return sum(r.decision for r in absent) / len(absent)

    def d_prime(self) -> float:
        """Signal-detection-theory sensitivity index d'.

        Returns *nan* if the hit or false-alarm rate is 0 or 1 (boundary cases).
        """
        hr = self.hit_rate()
        far = self.false_alarm_rate()
        if any(np.isnan(v) for v in (hr, far)):
            return float("nan")
        # Loglinear correction to avoid infinite z-scores
        n_present = sum(1 for r in self.trial_results if r.trial.target_present)
        n_absent = len(self.trial_results) - n_present
        hr_adj = (sum(r.decision for r in self.trial_results if r.trial.target_present) + 0.5) / (
            n_present + 1
        )
        far_adj = (sum(r.decision for r in self.trial_results if not r.trial.target_present) + 0.5) / (
            n_absent + 1
        )
        from scipy.stats import norm  # type: ignore[import]

        return float(norm.ppf(hr_adj) - norm.ppf(far_adj))

    def search_efficiency(
        self,
        random_mean_fixations: Optional[float] = None,
    ) -> float:
        """Search efficiency relative to a random-search baseline.

        Defined as ``random_mean_fixations / mean_fixations``.  Values > 1
        indicate better-than-random performance.  If *random_mean_fixations*
        is not provided, returns the reciprocal of mean fixations.

        Parameters
        ----------
        random_mean_fixations:
            Expected mean fixations for a random strategy (optional).
        """
        mf = self.mean_fixations
        if np.isnan(mf) or mf == 0:
            return float("nan")
        if random_mean_fixations is None:
            return 1.0 / mf
        return random_mean_fixations / mf

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Model           : {self.model_name}",
            f"Trials          : {len(self.trial_results)}",
            f"Accuracy        : {self.accuracy:.3f}",
            f"Hit rate        : {self.hit_rate():.3f}",
            f"False alarm rate: {self.false_alarm_rate():.3f}",
            f"Mean fixations  : {self.mean_fixations:.2f}",
            f"Mean time (s)   : {self.mean_elapsed_s:.4f}",
        ]
        try:
            dp = self.d_prime()
            lines.append(f"d'              : {dp:.3f}")
        except Exception:
            pass
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------


class Benchmark:
    """Run one or more :class:`ActiveVisionModel` instances on a set of trials.

    Parameters
    ----------
    max_fixations:
        Maximum number of fixations allowed per trial.
    fovea_radius:
        Radius (pixels) of the full-resolution foveal region.
    periphery_sigma:
        Gaussian blur sigma for the peripheral region.
    verbose:
        If ``True``, print per-trial progress.

    Examples
    --------
    >>> task = VisualSearchTask(image_size=(64, 64), n_distractors=4, seed=0)
    >>> model = RandomSearchModel(seed=0)
    >>> bench = Benchmark(max_fixations=10)
    >>> result = bench.run(model, task.generate_dataset(n_trials=20))
    >>> 0.0 <= result.accuracy <= 1.0
    True
    """

    def __init__(
        self,
        max_fixations: int = 20,
        fovea_radius: int = 16,
        periphery_sigma: float = 4.0,
        verbose: bool = False,
    ) -> None:
        self.max_fixations = max_fixations
        self.fovea_radius = fovea_radius
        self.periphery_sigma = periphery_sigma
        self.verbose = verbose

    def run_trial(self, model: ActiveVisionModel, trial: Trial) -> TrialResult:
        """Run *model* on a single *trial* and return a :class:`TrialResult`."""
        t0 = time.perf_counter()
        model.reset(trial.image)
        fixations: List[Tuple[int, int]] = []

        for _ in range(self.max_fixations):
            fov_image = (
                foveate(trial.image, fixations[-1], self.fovea_radius, self.periphery_sigma)
                if fixations
                else trial.image.copy()
            )
            loc = model.next_fixation(fov_image, list(fixations))
            fixations.append(loc)

        decision = model.make_decision()
        elapsed = time.perf_counter() - t0
        correct = decision == trial.target_present

        if self.verbose:
            mark = "✓" if correct else "✗"
            print(
                f"  {mark} target={'present' if trial.target_present else 'absent':7s} "
                f"decision={'present' if decision else 'absent':7s} "
                f"fixations={len(fixations)} time={elapsed:.3f}s"
            )

        return TrialResult(
            trial=trial,
            fixations=fixations,
            decision=decision,
            correct=correct,
            elapsed_s=elapsed,
        )

    def run(
        self,
        model: ActiveVisionModel,
        trials: Sequence[Trial],
        model_name: Optional[str] = None,
    ) -> BenchmarkResult:
        """Evaluate *model* on *trials* and return a :class:`BenchmarkResult`.

        Parameters
        ----------
        model:
            An :class:`ActiveVisionModel` instance.
        trials:
            A sequence of :class:`Trial` objects (e.g. from
            :meth:`VisualSearchTask.generate_dataset`).
        model_name:
            Display name for the model in the summary.  Defaults to the class
            name.
        """
        if model_name is None:
            model_name = type(model).__name__
        results = [self.run_trial(model, t) for t in trials]
        return BenchmarkResult(trial_results=results, model_name=model_name)

    def compare(
        self,
        models: Dict[str, ActiveVisionModel],
        trials: Sequence[Trial],
    ) -> Dict[str, BenchmarkResult]:
        """Evaluate multiple models on the same trials.

        Parameters
        ----------
        models:
            Mapping from display name to model instance.
        trials:
            Shared list of trials.

        Returns
        -------
        dict[str, BenchmarkResult]
            One :class:`BenchmarkResult` per model.
        """
        return {name: self.run(model, trials, model_name=name) for name, model in models.items()}


# ---------------------------------------------------------------------------
# Visualisation helpers (optional — requires matplotlib)
# ---------------------------------------------------------------------------


def plot_trial(
    trial_result: TrialResult,
    ax: Any = None,
    show_fixations: bool = True,
) -> Any:
    """Plot the stimulus with the scanpath overlaid.

    Parameters
    ----------
    trial_result:
        A completed :class:`TrialResult`.
    ax:
        Matplotlib :class:`~matplotlib.axes.Axes`.  Created if *None*.
    show_fixations:
        Whether to draw the fixation sequence.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plot_trial()") from exc

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(trial_result.trial.image, cmap="gray", origin="upper", vmin=0, vmax=1)

    if show_fixations and trial_result.fixations:
        rows = [f[0] for f in trial_result.fixations]
        cols = [f[1] for f in trial_result.fixations]
        ax.plot(cols, rows, "r-o", markersize=4, linewidth=1, alpha=0.7, label="Scanpath")
        ax.plot(cols[0], rows[0], "gs", markersize=8, label="Start")
        ax.plot(cols[-1], rows[-1], "r^", markersize=8, label="End")

    if trial_result.trial.target_location is not None:
        tr, tc = trial_result.trial.target_location
        ax.plot(tc, tr, "bx", markersize=12, markeredgewidth=2, label="Target")

    status = "correct" if trial_result.correct else "incorrect"
    decision = "present" if trial_result.decision else "absent"
    ax.set_title(f"Decision: {decision} ({status})")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_roc(
    results: Dict[str, BenchmarkResult],
    ax: Any = None,
) -> Any:
    """Plot hit-rate vs. false-alarm-rate for multiple models.

    Parameters
    ----------
    results:
        Mapping from model name to :class:`BenchmarkResult`.
    ax:
        Matplotlib axes.  Created if *None*.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plot_roc()") from exc

    if ax is None:
        _, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    for name, res in results.items():
        ax.plot(res.false_alarm_rate(), res.hit_rate(), "o", markersize=10, label=name)

    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Hit Rate")
    ax.set_title("ROC space")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return ax
