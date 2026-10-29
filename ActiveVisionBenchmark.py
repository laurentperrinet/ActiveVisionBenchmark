"""
ActiveVisionBenchmark
=====================

A lightweight, single-file toolkit to build controlled visual-flow stimuli for
active-vision experiments and to benchmark models that estimate or control
their own motion from those stimuli.

The library has three layers, all contained in this file:

1. **Stimulus layer** -- objects that create texture images (:class:`GridStimulus`,
   :class:`MotionCloudStimulus`, :class:`ImageStimulus`, ...).
2. **Task layer** -- objects that project a texture onto a 3D geometry seen by a
   pinhole camera and that define the stimulation protocol:
   :class:`CylinderTask` (a tunnel), :class:`HorizontalPlaneTask` (an infinite
   ground plane) and :class:`LandingTask` (a descent onto that plane).
3. **Benchmark layer** -- :class:`Benchmark` evaluates models on a set of trials.
   :class:`PassiveBenchmark` generates streams of images (open loop) and
   :class:`ActiveBenchmark` lets a model control the camera (closed loop).

Dependencies: ``numpy``, ``torch`` and ``torchvision`` are required.
``matplotlib``, ``imageio``, ``IPython`` (for ``show_video``) and ``MotionClouds`` are optional
and imported lazily.
Copying this file next to your script is a valid way to install it.

Model
-----

A pinhole camera looks at the world with the azimuth :math:`u` (horizontal) and
the elevation :math:`v` of each pixel measured from its optical axis (``gaze``).
The camera translates along the direction ``heading``, so that the focus of
expansion (FoE) of the optic flow is seen at the azimuth

.. math:: u_\\mathrm{FoE} = h - g,

where :math:`h` is the heading and :math:`g` the gaze (both in degrees, measured in
the world).

*Tunnel* (cylinder of radius :math:`R`, texture period :math:`P`): the relative angle
of a pixel around the axis and its deviation from the axis are

.. math:: \\theta = \\arctan\\frac{v}{u - u_\\mathrm{FoE}}, \\qquad
          \\phi \\approx \\sqrt{(u - u_\\mathrm{FoE})^2 + v^2},

and the depth of the wall seen in that pixel is :math:`Z = R / \\tan\\phi`. The texture
is sampled at :math:`(\\theta/\\pi,\\; (Z + s) \\bmod P)` where :math:`s` is the distance
travelled, and its luminance is attenuated by :math:`(Z_\\mathrm{min}/Z)^{n}`. The
exponent :math:`n` is a property of the light. A headlight at the camera is a point source
and the irradiance of a Lambertian surface falls as :math:`1/d^2`, hence :math:`n = 2`
(the default). Extended sources such as the neon tubes of a corridor decay more slowly
(:math:`n \\approx 1`), and :math:`n = 0` removes the attenuation (the landing task).

*Plane* (ground at :math:`Y=0` seen from the height :math:`H` with the pitch
:math:`p`): a pixel with the ray :math:`\\mathbf d = (d_x, d_y, d_z)` (in the world frame)
hits the ground at :math:`t = H / (-d_y)` and the texture is sampled at the position
:math:`(X, Z) \\bmod P` of that point.

*Perturbation* (exogenous, in degrees or in m/s): a slow Langevin diffusion (always
present), in which a hidden drive :math:`y` relaxes under a white random force
:math:`w` and is integrated by a damped signal :math:`x`,

.. math:: \\tau\\,\\dot y = -y + w(t), \\qquad \\tau\\,\\dot x = y - x,

whose stationary standard deviation is :math:`\\sigma_\\mathrm{OU}`, plus abrupt Gaussian
jumps of standard deviation :math:`\\sigma_\\mathrm{jump}` that arrive with a hazard rate
:math:`\\lambda` (only in the perturbation benchmarks).

*Time* : the camera moves with the time step :math:`\\Delta t` (``dt``), the frame rate of the
sensor. The model may be slower than the sensor: it decides only every :math:`k` frames
(``control_every``), so that the time step of the whole perception-action loop is
:math:`k\\,\\Delta t` (``control_dt``). In between, the camera keeps moving with the last
command (a zero-order hold), a gaze shift being applied once, and the model sees two
frames that are :math:`k` steps apart.

=======================  ==================  ========================  ==============================
Symbol                   Code                Default                   Role
=======================  ==================  ========================  ==============================
:math:`VA`               ``visual_angle``    45 deg                    visual angle of the image width
:math:`\\Delta t`         ``dt``              1/60 s (1/30 landing)     time step of the camera
:math:`k`                ``control_every``   1                         camera steps per model step
:math:`R`                ``radius``          1 m                       radius of the tunnel
:math:`P`                ``period``          10 m                      period of the texture
:math:`n`                ``depth_exponent``  2 (0 landing)             luminance attenuation exponent
:math:`H`                ``camera_height``   1.5 m                     height of the camera (plane)
:math:`\\tau`             ``tau``             1 s                       time constant of the diffusion
:math:`\\sigma_{OU}`      ``sigma_ou``        1 deg plane, 2 deg        amplitude of the diffusion
                                              tunnel, 0.1 m/s landing
:math:`\\lambda`          ``hazard_rate``     0 s^-1 (0.2 landing)      hazard rate of the jumps
:math:`\\sigma_{jump}`    ``sigma_jump``      0 (0.3 m/s landing)       standard deviation of a jump
=======================  ==================  ========================  ==============================

All angles are stored in degrees and converted to radians when needed.
"""

from __future__ import annotations

import os

# Operators missing on the Metal backend fall back to the CPU. This has to be
# set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import copy
import dataclasses
import datetime
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import pickle
import statistics
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "0.2.0"
__author__ = "Laurent Perrinet"
__license__ = "GPL-3.0-or-later"

# ---------------------------------------------------------------------------
# Constants and experiment utilities
# ---------------------------------------------------------------------------

#: Size reduction factor for quick tests (``AVB_DEBUG=10`` divides sizes by 10).
#: It is 1 in production.
DEBUG: int = max(1, int(os.environ.get("AVB_DEBUG", "1")))

#: Switch of the figures and videos of the notebooks. It is off by default, so that running
#: a notebook writes nothing outside of its own outputs. It is turned on with
#: ``AVB_FIGURES=1`` only when the figures are needed, that is when the paper is compiled.
SAVE_FIGURES: bool = os.environ.get("AVB_FIGURES", "0").strip().lower() in ("1", "true", "yes", "on")

#: Conversion from centimetres to the inches expected by matplotlib's ``figsize``.
CM: float = 1 / 2.54

#: The golden ratio, the default aspect ratio of the figures.
GOLDEN: float = (1 + 5**0.5) / 2


def scaled(n: int, minimum: int = 1) -> int:
    """Return *n* reduced by the :data:`DEBUG` factor.

    Args:
        n: Production value (number of trials, steps, ...).
        minimum: Smallest value returned.
    """
    return max(minimum, int(n) // DEBUG)


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Return the compute device: Metal (``mps``), then CUDA, then the CPU.

    Args:
        prefer: Name of a device (``"cpu"``, ``"mps"``, ...) that overrides the
            automatic choice.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` from *seed*."""
    return np.random.default_rng(seed)


def _json_default(obj: Any) -> Any:
    """Make parameter objects (stimuli, arrays, dataclasses) hashable by JSON."""
    if hasattr(obj, "params") and isinstance(obj.params, dict):
        return {"__class__": type(obj).__name__, **obj.params}
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"__class__": type(obj).__name__, **dataclasses.asdict(obj)}
    return repr(obj)


def _source_digest() -> str:
    """Short hash of this file, ``"unknown"`` if the source is not readable."""
    try:
        with open(__file__, "rb") as fh:
            return hashlib.sha1(fh.read()).hexdigest()[:10]
    except OSError:  # pragma: no cover
        return "unknown"


_SOURCE_DIGEST = _source_digest()


def params_hash(params: Any, length: int = 10) -> str:
    """Return a short hash that identifies a set of parameters.

    Args:
        params: Any JSON-serialisable structure (stimuli and dataclasses are
            handled).
        length: Number of hexadecimal characters kept.
    """
    text = json.dumps(params, sort_keys=True, default=_json_default)
    return hashlib.sha1(text.encode()).hexdigest()[:length]


def cached(
    name: str,
    params: Any,
    fn: Callable[[], Any],
    recompute: bool = False,
    cache_dir: str = "cache",
) -> Any:
    """Compute ``fn()`` once and store the result on disk.

    The cache file is named ``{ISO date}_{name}_{hash of the parameters}.pkl.gz``.
    The hash covers *params*, the library version and a digest of this file, so that
    any parameter that changes the result must be listed in *params*. The newest
    file with the same name and hash is reused, whatever its date.

    Args:
        name: Short name of the computation.
        params: Every parameter that influences the result.
        fn: Zero-argument callable that computes the result.
        recompute: If ``True``, ignore the cache and overwrite it.
        cache_dir: Directory of the cache files (created if needed).

    Returns:
        The (possibly cached) result of ``fn()``.
    """
    # The code is part of the key: changing the library invalidates the cached results.
    key = params_hash({"version": __version__, "source": _SOURCE_DIGEST, "params": params})
    suffix = f"_{name}_{key}.pkl.gz"
    os.makedirs(cache_dir, exist_ok=True)
    existing = sorted(f for f in os.listdir(cache_dir) if f.endswith(suffix))
    if existing and not recompute:
        with gzip.open(os.path.join(cache_dir, existing[-1]), "rb") as fh:
            return pickle.load(fh)
    result = fn()
    path = os.path.join(cache_dir, datetime.date.today().isoformat() + suffix)
    # Atomic write: a crash never leaves a truncated cache file behind.
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return result


def parallel_map(fn: Callable[[Any], Any], items: Iterable[Any], n_jobs: Optional[int] = None) -> List[Any]:
    """Apply *fn* to every item on all CPU cores.

    Child processes are started with ``spawn``: they never reuse the Metal
    context of the parent, which makes them safe on Apple silicon.

    Args:
        fn: Function importable from a module (a function defined in a notebook
            cell cannot be pickled by ``spawn``).
        items: Arguments of the calls.
        n_jobs: Number of processes. Defaults to ``os.cpu_count()``.
    """
    items = list(items)
    n = n_jobs if n_jobs is not None else (os.cpu_count() or 1)
    n = min(n, len(items))
    if n <= 1:
        return [fn(item) for item in items]
    try:
        pool = ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"))
    except (PermissionError, OSError) as exc:  # e.g. inside a sandbox that forbids semaphores
        warnings.warn(f"multiprocessing is unavailable ({exc}); running on a single core", stacklevel=2)
        return [fn(item) for item in items]
    with pool:
        return list(pool.map(fn, items, chunksize=max(1, len(items) // (4 * n))))


def new_figure(
    width_cm: float = 12.0,
    height_cm: Optional[float] = None,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Create a matplotlib figure sized in centimetres, in golden-ratio proportions.

    Args:
        width_cm: Width of the figure (cm).
        height_cm: Height of the figure (cm). Defaults to ``width_cm / GOLDEN``.
        nrows: Number of rows of axes.
        ncols: Number of columns of axes.
        **kwargs: Passed to :func:`matplotlib.pyplot.subplots`.

    Returns:
        ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt  # lazy: matplotlib is optional

    if height_cm is None:
        height_cm = width_cm / GOLDEN
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.titlesize": 8})
    kwargs.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=(width_cm * CM, height_cm * CM), **kwargs)


def save_figure(
    fig: Any,
    name: str,
    directory: str = "figures",
    formats: Sequence[str] = ("pdf", "png", "svg"),
    dpi: int = 300,
    force: bool = False,
) -> List[str]:
    """Save a figure with a transparent background in several formats.

    Nothing is written unless the switch :data:`SAVE_FIGURES` is on (``AVB_FIGURES=1``).

    Args:
        fig: Matplotlib figure.
        name: File name without extension.
        directory: Output directory (created if needed).
        formats: File extensions to write.
        dpi: Resolution of the raster formats (dots per inch).
        force: If ``True``, write the figure whatever the switch.

    Returns:
        The list of written paths (empty when the switch is off).
    """
    if not (SAVE_FIGURES or force):
        return []
    os.makedirs(directory, exist_ok=True)
    paths = []
    for ext in formats:
        path = os.path.join(directory, f"{name}.{ext}")
        fig.savefig(path, transparent=True, dpi=dpi)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Stimulus layer
# ---------------------------------------------------------------------------


class Stimulus:
    """Base class of the objects that create texture images.

    A stimulus is called with ``stimulus(size, seed)`` and returns a ``float32``
    array in [0, 1] of shape ``(height, width)`` (grey levels) or
    ``(height, width, 3)`` (colour). The result is deterministic for a given seed.
    The parameters that define the stimulus are available as :attr:`params`,
    which makes them usable in the key of a cache.
    """

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters that define the stimulus."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def __call__(self, size: Tuple[int, int], seed: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


class GridStimulus(Stimulus):
    """A regular grid, for baseline experiments and control displays.

    Horizontal lines are blue and vertical lines are red; the bottom-left and
    top-right quadrants are tinted green and red; the central cells are dimmed to
    mark the fovea. In grey levels the same image is averaged over the colour
    channels.

    Args:
        spacing: Distance between two lines (pixels).
        colour: If ``True``, return an RGB texture.
        fovea_cells: Half-width of the dimmed central region (in cells).
    """

    def __init__(self, spacing: int = 32, colour: bool = False, fovea_cells: int = 3) -> None:
        self.spacing = int(spacing)
        self.colour = bool(colour)
        self.fovea_cells = int(fovea_cells)

    def __call__(self, size: Tuple[int, int], seed: Optional[int] = None) -> np.ndarray:
        h, w = size
        s = self.spacing
        img = np.ones((h, w, 3), dtype=np.float32)
        img[::s, :, 0:2] = 0.0  # horizontal lines are blue
        img[:, ::s, 1:3] = 0.0  # vertical lines are red
        img[h // 2:, : w // 2, 0] = 0.0
        img[h // 2:, : w // 2, 2] = 0.0  # bottom-left quadrant: green
        img[: h // 2, w // 2:, 1] = 0.0
        img[: h // 2, w // 2:, 2] = 0.0  # top-right quadrant: red
        r = self.fovea_cells * s
        img[h // 2 - r: h // 2 + r, w // 2 - r: w // 2 + r] *= 0.5
        return img if self.colour else img.mean(axis=-1)


class MotionCloudStimulus(Stimulus):
    """A static Motion Cloud: band-pass, random-phase noise, periodic by construction.

    The texture is a log-Gabor band-pass noise centred on the spatial frequency
    ``sf_0`` with a bandwidth ``B_sf`` (both in cycles per pixel), rectified to a
    mean of 0.5 and a maximum Michelson contrast of 1.

    The ``MotionClouds`` package (optional dependency) is used when it is installed
    (``backend="auto"``); otherwise a built-in numpy implementation of the same
    envelope is used. Both backends share their statistics but **not their pixels**
    for a given seed. :attr:`params` records the backend that is actually used.

    Args:
        sf_0: Central spatial frequency (cycles per pixel).
        B_sf: Bandwidth of the spatial frequency (cycles per pixel).
        B_theta: Orientation bandwidth (rad). ``inf`` is isotropic.
        theta: Mean orientation (rad), used if ``B_theta`` is finite.
        backend: ``"auto"``, ``"motionclouds"`` or ``"builtin"``.
    """

    def __init__(
        self,
        sf_0: float = 0.125,
        B_sf: float = 0.1,
        B_theta: float = float("inf"),
        theta: float = 0.0,
        backend: str = "auto",
    ) -> None:
        if backend not in ("auto", "motionclouds", "builtin"):
            raise ValueError(f"unknown backend {backend!r}")
        self.sf_0 = float(sf_0)
        self.B_sf = float(B_sf)
        self.B_theta = float(B_theta)
        self.theta = float(theta)
        self.backend = backend

    @property
    def backend_used(self) -> str:
        """The backend that is actually used (``"motionclouds"`` or ``"builtin"``)."""
        if self.backend == "builtin":
            return "builtin"
        try:
            import MotionClouds  # noqa: F401
        except ImportError:
            if self.backend == "motionclouds":
                raise ImportError("backend='motionclouds' requires: pip install MotionClouds") from None
            return "builtin"
        return "motionclouds"

    @property
    def params(self) -> Dict[str, Any]:
        return {**super().params, "backend_used": self.backend_used}

    def __call__(self, size: Tuple[int, int], seed: Optional[int] = None) -> np.ndarray:
        h, w = size
        if self.backend_used == "motionclouds":
            import MotionClouds as mc

            fx, fy, ft = mc.get_grids(w, h, 1)
            env = mc.envelope_gabor(fx, fy, ft, sf_0=self.sf_0, B_sf=self.B_sf,
                                    theta=self.theta, B_theta=self.B_theta)
            # MotionClouds draws its phases from the global numpy generator.
            z = mc.rectif(mc.random_cloud(env, seed=seed))[:, :, 0].T
            return np.ascontiguousarray(z, dtype=np.float32)
        return self._builtin(h, w, seed)

    def _builtin(self, h: int, w: int, seed: Optional[int]) -> np.ndarray:
        rng = _rng(seed)
        fx = np.fft.fftfreq(w)[None, :]
        fy = np.fft.fftfreq(h)[:, None]
        f = np.sqrt(fx**2 + fy**2)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Log-Gabor radial envelope, as in MotionClouds (DC component removed).
            radial = np.exp(-0.5 * np.log(f / self.sf_0) ** 2 / np.log((self.sf_0 + self.B_sf) / self.sf_0) ** 2) / f
            env = np.where(f > 0, radial, 0.0)
        if np.isfinite(self.B_theta):
            angle = np.arctan2(fy, fx)
            env = env * np.exp(np.cos(2 * (angle - self.theta)) / 4 / self.B_theta**2)
        spectrum = np.exp(2j * np.pi * rng.uniform(size=(h, w))) * rng.standard_normal((h, w)) * env
        z = np.fft.ifft2(spectrum).real
        z -= z.mean()
        return (0.5 * z / np.max(np.abs(z)) + 0.5).astype(np.float32)


class GaussianNoiseStimulus(Stimulus):
    """White Gaussian noise clipped to [0, 1].

    Args:
        mean: Mean luminance (0-1).
        std: Standard deviation of the luminance (0-1).

    Examples:
        >>> GaussianNoiseStimulus()((4, 6), seed=0).shape
        (4, 6)
    """

    def __init__(self, mean: float = 0.5, std: float = 0.1) -> None:
        self.mean = float(mean)
        self.std = float(std)

    def __call__(self, size: Tuple[int, int], seed: Optional[int] = None) -> np.ndarray:
        return np.clip(_rng(seed).normal(self.mean, self.std, size=size), 0.0, 1.0).astype(np.float32)


class BinaryNoiseStimulus(Stimulus):
    """White binary noise: every pixel is ``high`` or ``low``, independently.

    Args:
        low: Luminance of the dark pixels (0-1).
        high: Luminance of the bright pixels (0-1).
        p: Probability that a pixel is bright (0-1).

    Examples:
        >>> img = BinaryNoiseStimulus()((8, 8), seed=0)
        >>> sorted(set(img.ravel().tolist()))
        [0.0, 1.0]
    """

    def __init__(self, low: float = 0.0, high: float = 1.0, p: float = 0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.low = float(low)
        self.high = float(high)
        self.p = float(p)

    def __call__(self, size: Tuple[int, int], seed: Optional[int] = None) -> np.ndarray:
        return np.where(_rng(seed).uniform(size=size) < self.p, self.high, self.low).astype(np.float32)


class DotNoiseStimulus(Stimulus):
    """Sparse random dots: bright dots of diameter ``dot_size`` on a dark background.

    The dot centres are drawn independently and every dot is stamped as a disc.
    Their density is set so that the expected fraction of bright pixels is
    ``sparseness``, overlaps included. With ``dot_size = 1`` this is a
    :class:`BinaryNoiseStimulus` with ``p = sparseness``. The stamping wraps around
    the borders, so that the texture stays periodic as a :class:`Task` requires.

    Args:
        sparseness: Expected fraction of the pixels covered by a dot (0-1).
        dot_size: Diameter of a dot (pixels).
        low: Luminance of the background (0-1).
        high: Luminance of the dots (0-1).

    Examples:
        >>> img = DotNoiseStimulus(sparseness=0.1, dot_size=3)((16, 16), seed=0)
        >>> img.shape, float(img.max())
        ((16, 16), 1.0)
    """

    def __init__(self, sparseness: float = 0.05, dot_size: int = 1, low: float = 0.0, high: float = 1.0) -> None:
        if not 0.0 <= sparseness <= 1.0:
            raise ValueError(f"sparseness must be in [0, 1], got {sparseness}")
        if dot_size < 1:
            raise ValueError(f"dot_size must be at least 1 pixel, got {dot_size}")
        self.sparseness = float(sparseness)
        self.dot_size = int(dot_size)
        self.low = float(low)
        self.high = float(high)

    def __call__(self, size: Tuple[int, int], seed: Optional[int] = None) -> np.ndarray:
        r = (self.dot_size - 1) / 2
        reach = int(math.ceil(r))
        disc = [(dy, dx) for dy in range(-reach, reach + 1) for dx in range(-reach, reach + 1) if dy**2 + dx**2 <= r**2]
        # P(a pixel is covered) = 1 - (1 - p)^len(disc) for independent centres: solve it for p.
        p = 1.0 - (1.0 - self.sparseness) ** (1.0 / len(disc))
        centres = _rng(seed).uniform(size=size) < p
        covered = np.zeros(size, dtype=bool)
        for dy, dx in disc:
            covered |= np.roll(centres, (dy, dx), axis=(0, 1))
        return np.where(covered, self.high, self.low).astype(np.float32)


def _tiling_lattice(angle: float, cells: float) -> Tuple[int, int, float, float]:
    """Choose a rotation of the tiling that keeps the image periodic in both directions.

    The tiling repeats on the lattice of the integers. Turned by ``theta``, it repeats
    on the turned lattice, and the image (``cells`` periods wide) wraps around without
    a seam only if ``cells * cos(theta)`` and ``cells * sin(theta)`` are integers. This
    holds when ``tan(theta) = n / m`` is rational and ``cells`` is a multiple of
    ``sqrt(m**2 + n**2)``, which is not an integer in general. Among the pairs
    ``(m, n)`` whose period fits in the image, the closest angle is kept.

    Args:
        angle: Requested rotation (deg). The tiling is invariant by 90 deg, so only
            ``angle`` modulo 90 deg matters.
        cells: Requested number of periods along the width.

    Returns:
        ``(m, n, angle, cells)``: the integers with ``tan(theta) = n / m``, the rotation
        that is drawn (deg, in [0, 90]) and the number of periods along the width, the
        multiple of ``sqrt(m**2 + n**2)`` that is the closest to ``cells``.

    Examples:
        >>> _tiling_lattice(0.0, 4)
        (1, 0, 0.0, 4.0)
        >>> m, n, angle, cells = _tiling_lattice(45.0, 4)
        >>> (m, n, round(angle, 6), round(cells, 6))
        (1, 1, 45.0, 4.242641)
    """
    theta = math.radians(angle % 90.0)
    limit = int(math.floor(cells))
    mm, nn = np.meshgrid(np.arange(limit + 1), np.arange(limit + 1), indexing="ij")
    allowed = (np.gcd(mm, nn) == 1) & (mm * mm + nn * nn <= cells * cells)  # one period must fit in the image
    error = np.where(allowed, np.abs(np.arctan2(nn, mm) - theta), np.inf)
    m, n = (int(i) for i in np.unravel_index(np.argmin(error), error.shape))
    unit = math.hypot(m, n)  # smallest period of the image along its axes (period of the tiling)
    return m, n, math.degrees(math.atan2(n, m)), max(1, int(round(cells / unit))) * unit


def _mosque_tiling(size: Tuple[int, int], cells: float, colour: bool, angle: float = 0.0) -> np.ndarray:
    """Draw a periodic 4.8.8 tiling with an eight-pointed star in every octagon.

    In the unit of the period, the octagons (apothem 1/2) are centred on the
    integers and the small squares, turned by 45 deg, on the half-integers. A star
    is the union of two squares turned by 45 deg from each other. All the edges are
    anti-aliased over one pixel from their signed distance.

    Args:
        size: ``(height, width)`` of the image (pixels).
        cells: Number of periods along the width. The period along the height is
            the same, rounded to a whole number of the periods that wrap around
            (see :func:`_tiling_lattice`).
        colour: If ``True``, return an RGB image, otherwise the mean of the channels.
        angle: Rotation of the tiling (deg). The coordinates are turned, not the image,
            so that the result stays periodic: the angle and ``cells`` are replaced by
            the closest values that wrap around (see :func:`_tiling_lattice`).

    Returns:
        ``float32`` array in [0, 1] that wraps around without a seam.
    """
    h, w = size
    m, n, _, cells = _tiling_lattice(angle, cells)
    unit = math.hypot(m, n)
    ny = max(1, int(round(cells * h / w / unit))) * unit
    cos, sin = m / unit, n / unit  # exact, so that a period of the image is a period of the turned lattice
    x = ((np.arange(w) + 0.5) / w * cells)[None, :]
    y = ((np.arange(h) + 0.5) / h * ny)[:, None]
    xr, yr = cos * x + sin * y, cos * y - sin * x  # position in the frame of the tiling
    ax = np.abs(xr - np.round(xr))  # position in the octagon that is the nearest
    ay = np.abs(yr - np.round(yr))
    pixel = cells / w  # one pixel (period)

    def cover(distance: np.ndarray) -> np.ndarray:
        return np.clip(0.5 - distance / pixel, 0.0, 1.0)

    def lerp(base: np.ndarray, colour_: Tuple[float, float, float], weight: np.ndarray) -> np.ndarray:
        return base + (np.asarray(colour_, dtype=np.float32) - base) * weight[..., None]

    def star(half_side: float) -> np.ndarray:
        return np.minimum(np.maximum(ax, ay) - half_side, (ax + ay) / math.sqrt(2.0) - half_side)

    d_octagon = np.maximum(np.maximum(ax, ay), (ax + ay) / math.sqrt(2.0)) - 0.5
    d_star, d_inner = star(0.32), star(0.14)
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:] = (0.07, 0.16, 0.50)  # cobalt: the small squares
    img = lerp(img, (0.94, 0.89, 0.75), cover(d_octagon))  # cream: the octagons
    img = lerp(img, (0.04, 0.47, 0.52), cover(d_star))  # turquoise: the large stars
    img = lerp(img, (0.85, 0.60, 0.15), cover(d_inner))  # ochre: the small stars
    grout = np.maximum.reduce([cover(np.abs(d_octagon) - 0.015), cover(np.abs(d_star) - 0.009),
                               cover(np.abs(d_inner) - 0.006)])
    img = lerp(img, (0.10, 0.07, 0.05), grout)  # dark joints
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img if colour else img.mean(axis=-1)


class ImageStimulus(Stimulus):
    """An image read from a file (for instance a natural image), or a mosque tiling.

    Without a ``path`` the image is a periodic geometric tiling that is drawn by
    the library (see :func:`_mosque_tiling`), so that the file stays usable on its
    own and the texture wraps around without a seam. It does not depend on the seed.

    Args:
        path: Path of an image file readable by :func:`torchvision.io.read_image`.
            ``None`` draws the tiling.
        colour: If ``True``, keep the RGB channels. Otherwise average them.
        cells: Number of periods of the tiling along the width, used only if
            ``path`` is ``None``. The default, 13, is the period of the default angle.
        angle: Rotation of the tiling (deg), used only if ``path`` is ``None``. The
            default, 22.62 deg, is the angle of the 5-12-13 triangle (``tan = 5/12``),
            which wraps around exactly with 13 periods. The
            image must stay periodic in both directions, and a turned tiling wraps
            around only if the tangent of the angle is rational and ``cells`` is a
            multiple of the matching period. The angle is therefore replaced by the
            closest such angle whose period fits in ``cells`` (a larger ``cells``
            allows a finer angle), and ``cells`` by the closest multiple. The values
            that are drawn are :attr:`tiling_angle` and :attr:`tiling_cells`.

    Examples:
        >>> ImageStimulus()((64, 64)).shape
        (64, 64)
        >>> ImageStimulus(colour=True)((64, 64)).shape
        (64, 64, 3)
        >>> stimulus = ImageStimulus()
        >>> round(stimulus.tiling_angle, 2), stimulus.tiling_cells
        (22.62, 13.0)
        >>> stimulus = ImageStimulus(cells=4, angle=45.0)
        >>> stimulus.tiling_angle, round(stimulus.tiling_cells, 3)
        (45.0, 4.243)
    """

    def __init__(self, path: Optional[str] = None, colour: bool = False, cells: int = 13, angle: float = 22.62) -> None:
        if cells < 1:
            raise ValueError(f"cells must be at least 1, got {cells}")
        self.path = None if path is None else str(path)
        self.colour = bool(colour)
        self.cells = int(cells)
        self.angle = float(angle)

    @property
    def tiling_angle(self) -> float:
        """Rotation of the tiling that is drawn (deg), the closest one to ``angle`` that wraps around."""
        return _tiling_lattice(self.angle, self.cells)[2]

    @property
    def tiling_cells(self) -> float:
        """Number of periods of the tiling along the width that is drawn, the closest one to ``cells`` that wraps around."""
        return _tiling_lattice(self.angle, self.cells)[3]

    def __call__(self, size: Tuple[int, int], seed: Optional[int] = None) -> np.ndarray:
        if self.path is None:
            return _mosque_tiling(tuple(size), self.cells, self.colour, self.angle)
        from torchvision.io import ImageReadMode, read_image  # lazy: keeps the import light

        mode = ImageReadMode.RGB if self.colour else ImageReadMode.GRAY
        img = read_image(self.path, mode=mode).float() / 255.0  # (C, H, W)
        if tuple(img.shape[-2:]) != tuple(size):
            img = F.interpolate(img[None], size=tuple(size), mode="bilinear", antialias=True, align_corners=False)[0]
        arr = img.permute(1, 2, 0).numpy()
        return np.ascontiguousarray(arr[:, :, 0] if arr.shape[-1] == 1 else arr, dtype=np.float32)


class ArrayStimulus(Stimulus):
    """Wrap a texture that is already available as an array.

    Args:
        array: ``(H, W)``, ``(H, W, 1)`` or ``(H, W, 3)`` array or tensor in [0, 1].
    """

    def __init__(self, array: Any) -> None:
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[:, :, 0]
        if array.ndim not in (2, 3):
            raise ValueError(f"expected a 2-D or 3-D array, got shape {array.shape}")
        self._array = array
        self.shape = tuple(array.shape)
        self.digest = hashlib.sha1(array.tobytes()).hexdigest()[:10]

    def __call__(self, size: Tuple[int, int], seed: Optional[int] = None) -> np.ndarray:
        return self._array


def _as_stimulus(stimulus: Any) -> Stimulus:
    """Return *stimulus* if it is a :class:`Stimulus`, otherwise wrap it as an array."""
    return stimulus if isinstance(stimulus, Stimulus) else ArrayStimulus(stimulus)


# ---------------------------------------------------------------------------
# Exogenous perturbation of the heading (or of the speed)
# ---------------------------------------------------------------------------


@dataclass
class Perturbation:
    """Exogenous perturbation: a slow diffusion plus abrupt jumps.

    The signal is ``d(t) = mean + d_OU(t) + d_jumps(t)`` where

    * ``d_OU`` is a slow Langevin diffusion of time constant ``tau`` and
      stationary standard deviation ``sigma_ou``. A hidden drive ``y`` relaxes
      with the time constant ``tau`` under a white random force, and the
      signal ``x`` is the damped integral of ``y`` (``tau dy = -y dt + dW``,
      ``tau dx = (y - x) dt``), integrated with the Euler-Maruyama scheme. As in a
      damped limb driven by a command, ``x`` is smooth and differentiable, so that
      the heading does not jitter from frame to frame (which would create a spurious
      rotation of the optic flow). Its autocorrelation is
      ``(1 + |dt|/tau) exp(-|dt|/tau)``. It is always present and models the slow
      motion of the target (the FoE).
    * ``d_jumps`` is a sum of instantaneous steps that arrive at the rate
      ``hazard_rate`` with amplitudes drawn from a Gaussian of standard deviation
      ``sigma_jump``. It is only used by the perturbation benchmarks and is off
      by default.

    The two components are independent. The signal is in degrees when it
    perturbs the heading, and in m/s when it perturbs the speed.

    Args:
        dt: Time step (s).
        tau: Time constant of the diffusion (s).
        sigma_ou: Stationary standard deviation of the diffusion (deg or m/s).
        mean: Mean of the signal (deg or m/s).
        hazard_rate: Rate of the jumps (jumps per second). 0 disables them.
        sigma_jump: Standard deviation of the amplitude of a jump (deg or m/s).
        limit: If not ``None``, the signal is clipped to ``mean +/- limit``
            (deg or m/s).
        seed: Default seed of :meth:`sample`.
        jump_tau: If ``None`` (default), a jump is permanent, as a sudden change of
            heading. Otherwise the jumps relax to 0 with this time constant (s), as
            a gust that perturbs the speed.
    """

    dt: float = 1 / 60
    tau: float = 1.0
    sigma_ou: float = 1.0
    mean: float = 0.0
    hazard_rate: float = 0.0
    sigma_jump: float = 0.0
    limit: Optional[float] = None
    seed: Optional[int] = None
    jump_tau: Optional[float] = None

    def components(self, n_steps: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return the diffusion and the jumps as two arrays of length ``n_steps + 1``.

        Args:
            n_steps: Number of time steps.
            seed: Random seed. Defaults to :attr:`seed`.
        """
        rng = _rng(self.seed if seed is None else seed)
        ou = np.zeros(n_steps + 1)
        if self.sigma_ou > 0:
            s, tau, dt = self.sigma_ou, self.tau, self.dt
            kick = 2.0 * s * math.sqrt(dt / tau) * rng.standard_normal(n_steps)
            # Start from the stationary joint law of (x, y): Var(y) = 2 s^2 and
            # Cov(x, y) = Var(x) = s^2.
            y = s * math.sqrt(2.0) * rng.standard_normal()
            x = 0.5 * y + s * math.sqrt(0.5) * rng.standard_normal()
            ou[0] = x
            for k in range(n_steps):
                x, y = x + (y - x) * dt / tau, y - y * dt / tau + kick[k]
                ou[k + 1] = x
        jumps = np.zeros(n_steps + 1)
        if self.hazard_rate > 0 and self.sigma_jump > 0:
            happens = rng.random(n_steps) < 1.0 - math.exp(-self.hazard_rate * self.dt)
            steps = happens * rng.normal(0.0, self.sigma_jump, n_steps)
            if self.jump_tau is None:
                jumps[1:] = np.cumsum(steps)  # a jump is permanent
            else:
                decay = math.exp(-self.dt / self.jump_tau)  # a jump relaxes back (a gust)
                for k in range(n_steps):
                    jumps[k + 1] = jumps[k] * decay + steps[k]
        return ou, jumps

    def sample(self, n_steps: int, seed: Optional[int] = None) -> np.ndarray:
        """Return one trajectory ``d(t)`` of length ``n_steps + 1``.

        Args:
            n_steps: Number of time steps.
            seed: Random seed. Defaults to :attr:`seed`.
        """
        ou, jumps = self.components(n_steps, seed)
        d = self.mean + ou + jumps
        if self.limit is not None:
            d = np.clip(d, self.mean - self.limit, self.mean + self.limit)
        return d


# ---------------------------------------------------------------------------
# Image-based estimators (used by the models, and exposed for the notebooks)
# ---------------------------------------------------------------------------


def _gradients(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sobel gradients ``(d/dcol, d/drow)`` of a ``(H, W)`` image, in luminance per pixel."""
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=img.device) / 8.0
    x = F.pad(img[None, None], (1, 1, 1, 1), mode="replicate")
    return F.conv2d(x, kx[None, None])[0, 0], F.conv2d(x, kx.t()[None, None])[0, 0]


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of a ``(H, W)`` image (separable kernel, replicated borders)."""
    if sigma <= 0:
        return img
    radius = int(math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    y = F.pad(img[None, None], (radius, radius, radius, radius), mode="replicate")
    y = F.conv2d(y, kernel[None, None, None, :])
    return F.conv2d(y, kernel[None, None, :, None])[0, 0]


def _box(x: torch.Tensor, size: int) -> torch.Tensor:
    """Local mean over a ``size x size`` window of a ``(H, W)`` or ``(N, H, W)`` tensor."""
    y = x[None, None] if x.ndim == 2 else x[:, None]
    y = F.avg_pool2d(y, size, stride=1, padding=size // 2, count_include_pad=False)
    return y[0, 0] if x.ndim == 2 else y[:, 0]


def estimate_foe(
    previous: torch.Tensor,
    current: torch.Tensor,
    visual_angle: float,
    elevation: float = 0.0,
    window: int = 9,
    highpass: Optional[int] = 15,
    step: float = 0.5,
    rotation: float = 0.0,
    yaw_range: float = 0.2,
) -> float:
    """Estimate the azimuth of the focus of expansion from two successive frames.

    This is a *direct* method: no optic flow is computed. For a candidate FoE
    ``x0`` the flow of a forward translation is radial, ``f = s (p - x0)``, where the
    positive scalar ``s`` (the inverse depth, up to the speed) is assumed constant in
    a small window. The brightness constancy ``I_t + grad(I) . f = 0`` then gives
    the best ``s`` of each window in closed form, and the residual of all the
    windows is a cost that depends only on ``x0``. The FoE is the candidate with the
    lowest cost, refined by a parabola.

    The frames are first high-pass filtered, so that the slow variation of the
    luminance with the distance (a multiplicative gain that is not carried by the
    texture) does not bias the gradients. A uniform flow (the yaw) is fitted
    jointly, since the cost is polynomial in the FoE and in the yaw.

    Args:
        previous: Previous frame, ``(H, W)`` in [0, 1].
        current: Current frame, ``(H, W)`` in [0, 1].
        visual_angle: Visual angle of the image width (deg).
        elevation: Elevation of the FoE (deg); 0 for a camera that looks at the
            horizon.
        window: Size of the windows where the inverse depth is constant (pixels).
        highpass: Size of the window of the high-pass filter (pixels), or ``None``.
        step: Spacing of the candidate FoEs (deg).
        rotation: Rotation of the gaze to the right between the two frames (deg),
            known from the efference copy of the motor command. A yaw rotation
            shifts the whole image horizontally; it is removed from the previous
            frame so that only the expansion remains.
        yaw_range: Largest unknown yaw rotation per frame that is fitted jointly
            with the FoE (deg per frame). A rotation of the heading with respect to
            the scene (a yawing tunnel) adds a uniform horizontal flow, which would
            otherwise displace the FoE by ``-r / s``, large where the expansion is
            weak. 0 disables the fit.

    Returns:
        Azimuth of the FoE (deg) in the current image (relative to the current
        optical axis), or ``nan`` if the images carry no usable motion.
    """
    h, w = current.shape
    if rotation:
        # Content at the azimuth u moves to u - rotation: resample the previous frame there.
        xs = (torch.arange(w, device=current.device, dtype=current.dtype) + rotation * (w - 1) / visual_angle) / (w - 1) * 2 - 1
        grid = torch.stack((xs[None, :].expand(h, w), (torch.arange(h, device=current.device, dtype=current.dtype) / (h - 1) * 2 - 1)[:, None].expand(h, w)), dim=-1)[None]
        previous = F.grid_sample(previous[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
    if highpass:
        previous, current = previous - _box(previous, highpass), current - _box(current, highpass)
    mid, it = 0.5 * (previous + current), current - previous
    ix, iy = _gradients(mid)
    iy = -iy  # rows go down, the elevation goes up
    px_per_deg = (w - 1) / visual_angle
    x = (torch.arange(w, device=current.device) - (w - 1) / 2)[None, :]
    y = ((h - 1) / 2 - torch.arange(h, device=current.device))[:, None]
    a = ix * x + iy * (y - elevation * px_per_deg)  # g = a - ix * x0
    a2, a1, a0 = _box(a * a, window), _box(a * ix, window), _box(ix * ix, window)
    b1, b0, stt = _box(a * it, window), _box(ix * it, window), _box(it * it, window)

    # The windows overlap strongly: keeping one in ``window // 2`` is enough.
    sub = max(1, window // 2)
    a2, a1, a0, b1, b0, stt = (t[::sub, ::sub] for t in (a2, a1, a0, b1, b0, stt))

    half = 0.5 * visual_angle
    x0 = (torch.arange(-half, half + 1e-6, step, device=current.device) * px_per_deg)[:, None, None, None]
    n_yaw = 2 * int(round(yaw_range * px_per_deg / 0.05)) + 1 if yaw_range > 0 else 1
    yaw = torch.linspace(-yaw_range * px_per_deg, yaw_range * px_per_deg, n_yaw, device=current.device)
    yaw = yaw[None, :, None, None]  # uniform horizontal flow (pixels per frame)
    # With a uniform flow r added to the radial one, the brightness constancy
    # reads (I_t + r I_x) + s g = 0, so every window sum stays polynomial in (x0, r).
    sgg = a2 - 2 * a1 * x0 + a0 * x0**2
    sgt = (b1 - b0 * x0) + yaw * (a1 - a0 * x0)
    stt = stt + 2 * yaw * b0 + yaw**2 * a0
    eps = 1e-9 + 1e-4 * sgg.mean(dim=(-1, -2), keepdim=True)
    # Expansion only: a window with s < 0 (sgt > 0) does not explain anything.
    explained = torch.where(sgt < 0, sgt**2 / (sgg + eps), torch.zeros_like(sgt))
    cost = (stt - explained).sum(dim=(-1, -2))  # (candidates, yaws)
    if not torch.isfinite(cost).all() or float(cost.max() - cost.min()) <= 1e-12:
        return float("nan")
    i, j = divmod(int(torch.argmin(cost)), cost.shape[1])
    offset = 0.0
    if 0 < i < cost.shape[0] - 1:  # parabolic refinement of the minimum
        c0, c1, c2 = float(cost[i - 1, j]), float(cost[i, j]), float(cost[i + 1, j])
        curvature = c0 - 2 * c1 + c2
        if curvature > 0:
            offset = 0.5 * (c0 - c2) / curvature
    return float(-half + (i + offset) * step)


def estimate_divergence(
    previous: torch.Tensor,
    current: torch.Tensor,
    visual_angle: float,
    dt: float,
    radius_fraction: float = 0.9,
    smooth: float = 2.0,
) -> float:
    """Estimate the divergence rate of the optic flow around the image centre.

    For a flow ``f = k * alpha`` that expands from the centre (``alpha`` is the
    angular position), the brightness constancy ``I_t + grad(I) . f = 0`` gives the
    least-squares estimate ``k = -sum(I_t g) / sum(g^2)`` with
    ``g = grad(I) . alpha``. The divergence rate ``D = k / dt`` equals ``-v_z / z``
    for a camera that descends vertically onto a plane at the height ``z``.

    Args:
        previous: Previous frame, ``(H, W)`` in [0, 1].
        current: Current frame, ``(H, W)`` in [0, 1].
        visual_angle: Visual angle of the image width (deg).
        dt: Time between the two frames (s).
        radius_fraction: Fraction of the half visual angle used (a disc, to avoid
            the corners).
        smooth: Standard deviation (pixels) of the Gaussian blur applied to the
            frames. Close to the ground the texture is smooth and its gradient is
            as small as the noise: the noise of the gradients then inflates the
            denominator of the least-squares estimate and biases it towards 0
            (a factor 0.1 at 0.2 m for a noise of 0.01 without blur). Blurring
            removes the noise, which is at high frequencies, and keeps the signal.

    Returns:
        Divergence rate (s^-1), or ``nan`` if there is no usable texture.
    """
    h, w = current.shape
    previous, current = _blur(previous, smooth), _blur(current, smooth)
    mid, dif = 0.5 * (previous + current), current - previous
    ix, iy = _gradients(mid)
    rad_per_px = math.radians(visual_angle) / (w - 1)
    ix, iy = ix / rad_per_px, -iy / rad_per_px  # per radian, with v going up
    va_v = math.radians(visual_angle) * h / w
    x = torch.linspace(-math.radians(visual_angle) / 2, math.radians(visual_angle) / 2, w, device=current.device)[None, :]
    y = torch.linspace(va_v / 2, -va_v / 2, h, device=current.device)[:, None]
    g = ix * x + iy * y
    disc = (x**2 + y**2) <= (radius_fraction * math.radians(visual_angle) / 2) ** 2
    den = (g * g * disc).sum()
    if not torch.isfinite(den) or den <= 0:
        return float("nan")
    return float(-(dif * g * disc).sum() / den / dt)


def estimate_time_to_contact(
    previous: torch.Tensor,
    current: torch.Tensor,
    visual_angle: float,
    dt: float,
    **kwargs: Any,
) -> float:
    """Estimate the time to contact with the surface from two frames.

    A texture gives no depth, but the time to contact needs none: it is the inverse
    of the divergence of the optic flow, ``tau = z / (-v_z) = 1 / D`` (Nunes et
    al., 2023, ICCV; Izzo & de Croon, 2012). It is the time that is left before
    the contact along the path of the camera if the speed stays the same. It
    needs the visual angle of the camera to turn the displacement of the pixels
    into an angle.

    Args:
        previous: Previous frame, ``(H, W)`` in [0, 1].
        current: Current frame, ``(H, W)`` in [0, 1].
        visual_angle: Visual angle of the image width (deg).
        dt: Time between the two frames (s).
        **kwargs: Passed to :func:`estimate_divergence`.

    Returns:
        Time to contact (s): ``inf`` if the surface is not getting closer
        (divergence at most 0) and ``nan`` if there is no usable texture.

    Examples:
        >>> img = torch.rand(16, 16)
        >>> estimate_time_to_contact(img, img, 45.0, 1 / 30)
        inf
    """
    d = estimate_divergence(previous, current, visual_angle, dt, **kwargs)
    if not math.isfinite(d):
        return float("nan")
    return 1.0 / d if d > 0 else float("inf")


# ---------------------------------------------------------------------------
# State, observation and action
# ---------------------------------------------------------------------------


@dataclass
class Pose:
    """State of the camera.

    Attributes:
        distance: Distance travelled (m).
        gaze: Azimuth of the optical axis in the world (deg).
        heading: Azimuth of the direction of travel in the world (deg).
        altitude: Height above the ground (m), used by the landing task.
        x: World position, to the right (m), used by the plane tasks.
        z: World position, forward (m), used by the plane tasks.
    """

    distance: float = 0.0
    gaze: float = 0.0
    heading: float = 0.0
    altitude: float = 0.0
    x: float = 0.0
    z: float = 0.0

    @property
    def foe(self) -> float:
        """Azimuth of the focus of expansion in the image (deg)."""
        return self.heading - self.gaze


@dataclass
class Action:
    """What a model returns at each step. Every field is optional.

    Attributes:
        gaze_shift: Rotation of the gaze (deg), to the right.
        speed_command: Commanded speed (m/s), used by the landing task.
        heading_estimate: Estimate of the heading in the world (deg).
    """

    gaze_shift: float = 0.0
    speed_command: Optional[float] = None
    heading_estimate: Optional[float] = None


@dataclass
class Observation:
    """What a model sees at each step.

    Attributes:
        image: Current frame, ``(H, W)`` in [0, 1] (grey levels).
        previous: Frame of the previous decision of the model (``None`` at the
            first step). It is the previous camera frame if ``control_every`` is 1.
        gaze: Current gaze (deg).
        step: Index of the camera step.
        t: Time since the beginning of the episode (s).
        dt: Time between ``previous`` and ``image`` (s), that is the step of the
            perception-action loop, ``control_every * dt`` of the task.
        visual_angle: Visual angle of the image width (deg). It is a property of
            the camera that the models need to turn pixels into angles, for
            instance to convert a divergence into a time to contact.
        info: Ground-truth quantities for diagnostics and test controllers.
            Models that estimate their state from the images must not read it.
    """

    image: torch.Tensor
    previous: Optional[torch.Tensor]
    gaze: float
    step: int
    t: float
    dt: float
    visual_angle: float = 45.0
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trial:
    """A reproducible trial.

    Attributes:
        seed: Seed that determines the texture, the noise and the perturbation.
        target_present: Whether the heading is decentred with respect to the gaze
            (plane task).
        target_location: ``(row, col)`` of the FoE in the first image, or ``None``.
        heading_offset: Mean heading minus the initial gaze (deg).
        metadata: Arbitrary extra information.
    """

    seed: int = 0
    target_present: bool = False
    target_location: Optional[Tuple[int, int]] = None
    heading_offset: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Frame:
    """One frame of a passive stream.

    Attributes:
        image: ``(H, W)`` grey or ``(H, W, 3)`` colour image in [0, 1].
        pose: Pose of the camera.
        flow: Ground-truth optic flow ``(2, H, W)`` for the next step, in degrees
            per step (``flow[0]`` to the right, ``flow[1]`` upwards), or ``None``.
        t: Time (s).
        perturbation: Value of the exogenous perturbation (deg or m/s).
    """

    image: np.ndarray
    pose: Pose
    flow: Optional[np.ndarray]
    t: float
    perturbation: float


# ---------------------------------------------------------------------------
# Task layer
# ---------------------------------------------------------------------------


def _sample_periodic(texture: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """Sample a doubly periodic texture with bilinear interpolation.

    Args:
        texture: ``(1, C, H + 2, W + 2)`` texture padded circularly by one pixel,
            so that the interpolation is seamless across the period.
        gx: Horizontal coordinates, period 2 (one period is [-1, 1)).
        gy: Vertical coordinates, period 2.

    Returns:
        ``(C, h, w)`` sampled image.
    """
    h, w = texture.shape[2] - 2, texture.shape[3] - 2
    gx = torch.remainder(gx + 1.0, 2.0) - 1.0
    gy = torch.remainder(gy + 1.0, 2.0) - 1.0
    # The padded texture is (W + 2) wide for a period of W pixels.
    grid = torch.stack((gx * w / (w + 2), gy * h / (h + 2)), dim=-1)[None]
    return F.grid_sample(texture, grid, mode="bilinear", padding_mode="border", align_corners=False)[0]


class Task:
    """Base class of the tasks: a 3D geometry that is probed by a camera.

    A task projects a texture onto a surface (:meth:`render`), moves the camera at
    each time step (:meth:`step`) and gives the ground-truth optic flow
    (:meth:`flow`). It is used in open loop by :class:`PassiveBenchmark` and in
    closed loop by :class:`ActiveBenchmark`.

    Args:
        stimulus: A :class:`Stimulus`, or an array that is wrapped in an
            :class:`ArrayStimulus`.
        size: ``(height, width)`` of the images (pixels). An integer is a square.
        visual_angle: Visual angle of the image width (deg).
        dt: Duration of a time step of the camera (s), that is the inverse of its
            frame rate.
        speed: Speed of the camera (m/s).
        n_steps: Number of camera steps of an episode.
        perturbation: :class:`Perturbation` of the heading. Defaults to a slow
            diffusion without jumps.
        noise_std: Standard deviation of the Gaussian noise added to the
            observations (luminance, 0-1).
        texture_size: ``(height, width)`` of the texture (pixels).
        depth_exponent: Exponent ``n`` (at least 0) of the luminance attenuation with
            the distance, ``(Z_min / Z)^n``. It describes the light: 2 for a
            headlight at the camera (a point source, whose irradiance on a Lambertian
            surface falls as the inverse square of the distance), about 1 for an
            extended source such as neon tubes, whose decay is slower, and 0 for no
            attenuation.
        heading_lag: Time constant (s) with which the heading follows the gaze in
            closed loop. ``None`` means that the heading does not follow the gaze.
        warmup: Number of initial camera steps that are ignored by the metrics.
        seed: Master seed of :meth:`generate_dataset`.
        device: Compute device. By default, :func:`get_device` (Metal on Apple
            silicon) for images of at least 192 x 192 pixels and the CPU for
            smaller ones, which are faster there.
        control_every: Number of camera steps between two decisions of the model. A
            model that needs more time than ``dt`` to compute decides only every
            ``control_every`` steps and the time step of the whole perception-action
            loop is ``control_dt = control_every * dt``. In between, the camera keeps
            moving with the last command (a gaze shift is applied once) and the
            observation holds the frames of two successive decisions.
    """

    def __init__(
        self,
        stimulus: Any,
        size: Union[int, Tuple[int, int]] = 128,
        visual_angle: float = 45.0,
        dt: float = 1 / 60,
        speed: float = 2.0,
        n_steps: int = 300,
        perturbation: Optional[Perturbation] = None,
        noise_std: float = 0.0,
        texture_size: Union[int, Tuple[int, int]] = 256,
        depth_exponent: float = 2.0,
        heading_lag: Optional[float] = None,
        warmup: int = 0,
        seed: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        control_every: int = 1,
    ) -> None:
        if depth_exponent < 0:
            raise ValueError(f"depth_exponent must be at least 0, got {depth_exponent}")
        if control_every < 1 or int(control_every) != control_every:
            raise ValueError(f"control_every must be a whole number of steps, at least 1, got {control_every}")
        self.seed = int(seed)
        self.control_every = int(control_every)
        self.stimulus = _as_stimulus(stimulus)
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.visual_angle = float(visual_angle)
        self.dt = float(dt)
        self.speed = float(speed)
        self.n_steps = int(n_steps)
        self.perturbation = perturbation if perturbation is not None else Perturbation(dt=dt)
        self.noise_std = float(noise_std)
        self.texture_size = (texture_size, texture_size) if isinstance(texture_size, int) else tuple(texture_size)
        self.depth_exponent = float(depth_exponent)
        self.heading_lag = heading_lag
        self.warmup = int(warmup)
        if device is not None:
            self.device = torch.device(device)
        elif self.size[0] * self.size[1] >= 192 * 192:
            self.device = get_device()
        else:
            # For small images the launch overhead of the Metal kernels and the
            # synchronisations of the estimators dominate: the CPU is 10x faster
            # (measured on 128 x 128 episodes) and is what the worker processes use.
            self.device = torch.device("cpu")
        self.gaze_limit = 60.0  # the gaze is kept in a plausible range (deg)
        self._texture: Optional[torch.Tensor] = None
        self._uv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._closed_loop = True
        self._d = np.zeros(self.n_steps + 1)
        self._k = 0
        self._held = Action()  # last command of the model, held between two decisions
        self.pose = Pose()

    # -- pickling: device tensors are rebuilt after unpickling ------------------

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_texture"] = None
        state["_uv"] = None
        return state

    def to(self, device: Union[str, torch.device]) -> "Task":
        """Move the task to another device and return it."""
        self.device = torch.device(device)
        self._texture, self._uv = None, None
        return self

    # -- time ----------------------------------------------------------------

    @property
    def control_dt(self) -> float:
        """Time step of the perception-action loop (s): ``control_every * dt``."""
        return self.control_every * self.dt

    @property
    def control_due(self) -> bool:
        """Whether the model decides at the current camera step.

        The model is asked for an action only when this is ``True``; at the other
        steps :meth:`step` ignores the action it receives and holds the last one.
        """
        return self._k % self.control_every == 0

    def _hold(self, action: Optional[Action]) -> Action:
        """Return the action to apply at this step: the new one at a decision, the held one otherwise.

        A gaze shift is a displacement, not a command, so that it is applied once.
        """
        if self.control_due:
            self._held = action if action is not None else Action()
            return self._held
        return dataclasses.replace(self._held, gaze_shift=0.0)

    # -- geometry ------------------------------------------------------------

    @property
    def pixels_per_degree(self) -> float:
        """Number of pixels per degree of visual angle."""
        return (self.size[1] - 1) / self.visual_angle

    def _angles(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Azimuth ``u`` and elevation ``v`` (deg) of every pixel, as ``(H, W)`` tensors."""
        if self._uv is None:
            h, w = self.size
            va_v = self.visual_angle * h / w
            u = torch.linspace(-self.visual_angle / 2, self.visual_angle / 2, w, device=self.device)
            v = torch.linspace(va_v / 2, -va_v / 2, h, device=self.device)
            vv, uu = torch.meshgrid(v, u, indexing="ij")
            self._uv = (uu, vv)
        return self._uv

    def _texture_coords(self, pose: Pose) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Texture coordinates ``(gx, gy)`` and luminance gain of every pixel."""
        raise NotImplementedError

    def flow(self, pose: Pose, ds: Optional[float] = None) -> torch.Tensor:
        """Ground-truth optic flow for a forward translation of ``ds``.

        Args:
            pose: Pose of the camera.
            ds: Distance travelled (m). Defaults to one step, ``speed * dt``.

        Returns:
            ``(2, H, W)`` tensor of the displacement of every pixel, in degrees
            (``[0]`` to the right, ``[1]`` upwards). It is 0 outside the geometry.
        """
        raise NotImplementedError

    # -- texture and rendering -----------------------------------------------

    def prepare(self, seed: Optional[int] = 0) -> None:
        """Create the texture of a trial from its seed and send it to the device.

        Args:
            seed: Seed of the stimulus.
        """
        tex = np.asarray(self.stimulus(self.texture_size, seed), dtype=np.float32)
        tex = tex[None] if tex.ndim == 2 else np.moveaxis(tex, -1, 0)  # (C, H, W)
        padded = F.pad(torch.from_numpy(np.ascontiguousarray(tex))[None], (1, 1, 1, 1), mode="circular")
        self._texture = padded.to(self.device)

    def render(self, pose: Pose, seed: Optional[int] = None) -> torch.Tensor:
        """Render the image seen from *pose*.

        Args:
            pose: Pose of the camera.
            seed: If the texture has not been prepared yet, seed used to create it.

        Returns:
            ``(C, H, W)`` image in [0, 1] (``C`` is 1 for grey levels, 3 for colour).
        """
        if self._texture is None:
            self.prepare(0 if seed is None else seed)
        gx, gy, gain = self._texture_coords(pose)
        return (_sample_periodic(self._texture, gx, gy) * gain).clamp(0.0, 1.0)

    def _observe(self, pose: Pose, previous: Optional[torch.Tensor]) -> Observation:
        image = self.render(pose)
        if self.noise_std > 0:
            noise = torch.randn(image.shape, generator=self._noise_gen).to(self.device)
            image = (image + self.noise_std * noise).clamp(0.0, 1.0)
        grey = image.mean(dim=0)
        return Observation(
            image=grey,
            previous=previous,
            gaze=pose.gaze,
            step=self._k,
            t=self._k * self.dt,
            dt=self.control_dt,
            visual_angle=self.visual_angle,
            info={"foe": pose.foe, "heading": pose.heading, "gaze": pose.gaze, "altitude": pose.altitude},
        )

    # -- episodes ---------------------------------------------------------------

    def generate_trial(self, seed: int = 0) -> Trial:
        """Draw a trial from its seed."""
        return Trial(seed=seed)

    def generate_dataset(self, n_trials: int = 100, seed: Optional[int] = None) -> List[Trial]:
        """Generate ``n_trials`` reproducible trials.

        Args:
            n_trials: Number of trials.
            seed: Master seed. Trial ``i`` uses the seed ``seed * 100003 + i``.
                Defaults to the seed of the task.
        """
        seed = self.seed if seed is None else seed
        return [self.generate_trial(seed * 100003 + i) for i in range(n_trials)]

    def _initial_pose(self, trial: Trial) -> Pose:
        return Pose(gaze=0.0, heading=float(self._d[0]))

    def reset(
        self,
        trial: Optional[Trial] = None,
        seed: Optional[int] = None,
        closed_loop: bool = True,
    ) -> Observation:
        """Start an episode and return its first observation.

        Args:
            trial: The trial to play. If ``None``, a default trial with *seed*.
            seed: Seed used when no trial is given.
            closed_loop: If ``False``, the actions are ignored (open loop).
        """
        if trial is None:
            trial = self.generate_trial(0 if seed is None else seed)
        self._closed_loop = closed_loop
        self.prepare(trial.seed)
        self._noise_gen = torch.Generator().manual_seed(trial.seed + 2)
        pert = dataclasses.replace(self.perturbation, mean=trial.heading_offset)
        self._d = pert.sample(self.n_steps, seed=trial.seed + 1)
        self._k = 0
        self._held = Action()
        self.pose = self._initial_pose(trial)
        self._prev = None
        obs = self._observe(self.pose, None)
        self._prev = obs.image
        return obs

    def _advance(self, pose: Pose, action: Action, dd: float) -> Pose:
        """Return the pose after one step (to be refined by the subclasses)."""
        gaze, heading = pose.gaze, pose.heading + dd
        if self._closed_loop:
            gaze = float(np.clip(gaze + action.gaze_shift, -self.gaze_limit, self.gaze_limit))
            if self.heading_lag:
                heading += (gaze - heading) * self.dt / self.heading_lag
        ds = self.speed * self.dt
        h = math.radians(heading)
        return Pose(
            distance=pose.distance + ds, gaze=gaze, heading=heading, altitude=pose.altitude,
            x=pose.x + ds * math.sin(h), z=pose.z + ds * math.cos(h),
        )

    def _error(self, old: Pose, new: Pose, action: Action) -> float:
        """Task-specific error of one step (deg), ``nan`` if undefined."""
        return float("nan")

    def step(self, action: Optional[Action] = None) -> Tuple[Observation, bool, Dict[str, Any]]:
        """Advance the episode by one camera step.

        Args:
            action: Action of the model (ignored in open loop). It is read only
                when :attr:`control_due` is ``True``; at the other steps, the
                last action is held (see ``control_every``).

        Returns:
            ``(observation, done, info)``. ``info`` holds the pose before the step
            (``gaze_obs``, ``heading_obs``), ``error`` and, at the end of a
            landing, its outcome.
        """
        action = self._hold(action)
        old = self.pose
        dd = float(self._d[self._k + 1] - self._d[self._k])
        new = self._advance(old, action, dd)
        info: Dict[str, Any] = {
            "gaze_obs": old.gaze, "heading_obs": old.heading, "foe_obs": old.foe,
            "error": self._error(old, new, action),
        }
        self._k += 1
        self.pose = new
        obs = self._observe(new, self._prev)
        if self.control_due:  # the next decision compares its frame with this one
            self._prev = obs.image
        done = self._k >= self.n_steps
        return obs, done, info


class CylinderTask(Task):
    """Navigation along the axis of a tunnel (a textured cylinder).

    The observer always travels along the axis. The perturbation displaces its gaze,
    so that the axis, and with it the FoE, is seen away from the centre of the image.
    The task of the model is to keep its gaze on the axis, that is to bring the FoE
    back to the centre of the image. In the code the perturbation acts on
    ``Pose.heading``; only ``heading - gaze``, the azimuth of the FoE, matters.

    The texture is wrapped on the inner wall: its horizontal axis is the angle
    around the axis and its vertical axis is the depth, with the period ``period``.
    The observer is assumed to be on the axis (the tunnel is drawn as converging to
    the FoE), which is exact for an observer that travels along the axis.

    Args:
        stimulus: Texture, see :class:`Task`.
        size: ``(height, width)`` of the images (pixels).
        visual_angle: Visual angle of the image width (deg).
        period: Period of the texture along the axis (m).
        radius: Radius of the tunnel (m).
        mask: If ``True``, a circular mask of radius ``visual_angle / 2`` removes the
            border of the image. It is centred on the centre of the image (the
            screen), not on the FoE, so it stays fixed when the FoE is decentred.
        exact: If ``True``, use the exact great-circle distance to the axis
            instead of the small-angle approximation of the paper.
        heading_lag: Time constant (s) with which the heading is pulled towards the
            gaze in closed loop ("the observer moves where it looks"). By default
            (``None``) the heading follows its exogenous trajectory only, so that
            the model has to find the FoE with its gaze. With a short lag the
            heading follows the gaze by itself and the perturbation is absorbed,
            which makes the control trivial.
        **kwargs: Passed to :class:`Task` (``dt``, ``speed``, ``n_steps``,
            ``perturbation``, ``noise_std``, ``depth_exponent``, ...).
    """

    def __init__(
        self,
        stimulus: Any,
        size: Union[int, Tuple[int, int]] = 128,
        visual_angle: float = 45.0,
        period: float = 10.0,
        radius: float = 1.0,
        mask: bool = False,
        exact: bool = False,
        heading_lag: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("perturbation", Perturbation(dt=kwargs.get("dt", 1 / 60), tau=1.0, sigma_ou=2.0))
        # The expansion of a tunnel is weak near its axis, so the camera has to be fast
        # for the FoE to be visible in the flow of two successive frames.
        kwargs.setdefault("speed", 6.0)
        super().__init__(stimulus, size, visual_angle, heading_lag=heading_lag, **kwargs)
        self.period = float(period)
        self.radius = float(radius)
        self.mask = bool(mask)
        self.exact = bool(exact)

    def _phi_theta(self, pose: Pose) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        uu, vv = self._angles()
        u, v = torch.deg2rad(uu - pose.foe), torch.deg2rad(vv)
        if self.exact:
            phi = torch.acos((torch.cos(v) * torch.cos(u)).clamp(-1.0, 1.0))
            theta = torch.atan2(torch.sin(v), torch.cos(v) * torch.sin(u))
        else:
            phi, theta = torch.hypot(u, v), torch.atan2(v, u)
        return u, v, phi.clamp(min=1e-4), theta

    def _screen_mask(self) -> torch.Tensor:
        """Circular mask (1 inside, 0 outside) centred on the centre of the image, as an ``(H, W)`` tensor."""
        # Unlike phi, the angle to the centre of the screen does not depend on the pose:
        # the mask is a property of the display, not of the scene.
        uu, vv = self._angles()
        u, v = torch.deg2rad(uu), torch.deg2rad(vv)
        if self.exact:
            eccentricity = torch.acos((torch.cos(v) * torch.cos(u)).clamp(-1.0, 1.0))
        else:
            eccentricity = torch.hypot(u, v)
        return (eccentricity < math.radians(self.visual_angle / 2)).float()

    def _texture_coords(self, pose: Pose) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, phi, theta = self._phi_theta(pose)
        depth = self.radius / torch.tan(phi) + pose.distance % self.period
        gx = theta / math.pi
        gy = 2.0 * torch.remainder(depth, self.period) / self.period - 1.0
        phi_max = math.radians(self.visual_angle / 2)
        gain = ((torch.tan(phi) / math.tan(phi_max)) ** self.depth_exponent).clamp(max=1.0)
        if self.mask:
            gain = gain * self._screen_mask()
        return gx, gy, gain

    def flow(self, pose: Pose, ds: Optional[float] = None) -> torch.Tensor:
        ds = self.speed * self.dt if ds is None else ds
        u, v, phi, _ = self._phi_theta(pose)
        rho = torch.hypot(u, v).clamp(min=1e-4)
        depth = self.radius / torch.tan(phi)
        # A wall point at the depth Z gets closer by ds and moves away from the axis.
        phi_new = torch.atan(self.radius / (depth - ds).clamp(min=1e-3))
        d_phi = torch.rad2deg(phi_new - phi)
        flow = torch.stack((d_phi * u / rho, d_phi * v / rho))
        if self.mask:
            flow = flow * self._screen_mask()
        return flow

    def _error(self, old: Pose, new: Pose, action: Action) -> float:
        # Control objective: keep the gaze on the axis, i.e. the FoE at the centre.
        return abs(new.foe)


class HorizontalPlaneTask(Task):
    """Navigation over an infinite textured ground plane.

    The camera looks at the horizon (``pitch = 0``) from the height
    ``camera_height`` and translates along the heading, which is perturbed. The
    task of the model is to estimate the direction of the egomotion, that is the
    heading, as accurately as possible. Trials where the heading is decentred
    from the gaze by more than ``delta_min`` play the role of the *target present*
    trials of the signal-detection metrics.

    A pixel is a ray from the camera; where it hits the ground plane, the
    texture is sampled with the period ``period`` in both directions. The
    luminance is attenuated with the distance as ``(sin(depression))^n``, the
    sky being black.

    Args:
        stimulus: Texture, see :class:`Task`.
        size: ``(height, width)`` of the images (pixels).
        visual_angle: Visual angle of the image width (deg).
        camera_height: Height of the camera above the ground (m).
        pitch: Elevation of the optical axis (deg); negative looks down.
        period: Period of the texture on the ground (m).
        delta_min: Heading offset above which a trial is a *target present* trial,
            and decision threshold of the models (deg).
        h_max: Largest mean heading offset of a trial (deg).
        present_fraction: Fraction of the trials with a decentred heading.
        n_steps: Number of steps of a trial.
        speed: Speed of the camera (m/s).
        warmup: Steps ignored by the metrics (two frames are needed to see motion).
        **kwargs: Passed to :class:`Task`.
    """

    def __init__(
        self,
        stimulus: Any,
        size: Union[int, Tuple[int, int]] = 128,
        visual_angle: float = 45.0,
        camera_height: float = 1.5,
        pitch: float = 0.0,
        period: float = 10.0,
        delta_min: float = 2.0,
        h_max: float = 10.0,
        present_fraction: float = 0.5,
        n_steps: int = 30,
        speed: float = 4.0,
        warmup: int = 2,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("perturbation", Perturbation(dt=kwargs.get("dt", 1 / 60), tau=1.0, sigma_ou=1.0))
        super().__init__(stimulus, size, visual_angle, speed=speed, n_steps=n_steps, warmup=warmup, **kwargs)
        self.camera_height = float(camera_height)
        self.pitch = float(pitch)
        self.period = float(period)
        self.delta_min = float(delta_min)
        self.h_max = float(h_max)
        self.present_fraction = float(present_fraction)
        self._gain_ref: Optional[float] = None

    # -- camera model (overridden by the landing task) ----------------------

    def _camera_height(self, pose: Pose) -> float:
        return self.camera_height

    def _pitch(self, pose: Pose) -> float:
        return self.pitch

    def _displacement(self, pose: Pose, ds: float) -> Tuple[float, float, float]:
        """World displacement ``(dx, dy, dz)`` for a travelled distance ``ds``."""
        h = math.radians(pose.heading)
        return ds * math.sin(h), 0.0, ds * math.cos(h)

    def _rays(self, pose: Pose) -> Tuple[torch.Tensor, ...]:
        """Camera-frame ray and world-frame ray ``(x, y, z)`` of every pixel."""
        uu, vv = self._angles()
        u, v = torch.deg2rad(uu), torch.deg2rad(vv)
        cam = (torch.cos(v) * torch.sin(u), torch.sin(v), torch.cos(v) * torch.cos(u))
        p, g = math.radians(self._pitch(pose)), math.radians(pose.gaze)
        y1 = cam[1] * math.cos(p) + cam[2] * math.sin(p)
        z1 = -cam[1] * math.sin(p) + cam[2] * math.cos(p)
        world = (cam[0] * math.cos(g) + z1 * math.sin(g), y1, z1 * math.cos(g) - cam[0] * math.sin(g))
        return cam + world

    def _reference_gain(self, pose: Pose) -> float:
        """Largest luminance gain in the image (independent of the pose), for normalisation."""
        if self._gain_ref is None:
            ground = (-self._rays(pose)[4]).clamp(min=0.0)
            self._gain_ref = float((ground**self.depth_exponent).max()) or 1.0
        return self._gain_ref

    def _texture_coords(self, pose: Pose) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        *_, wx, wy, wz = self._rays(pose)
        below = wy < -1e-4
        t = self._camera_height(pose) / torch.where(below, -wy, torch.ones_like(wy))
        x = t * wx + pose.x % self.period
        z = t * wz + pose.z % self.period
        gx = 2.0 * torch.remainder(x, self.period) / self.period - 1.0
        gy = 2.0 * torch.remainder(z, self.period) / self.period - 1.0
        gain = (torch.where(below, -wy, torch.zeros_like(wy)) ** self.depth_exponent) / self._reference_gain(pose)
        return gx, gy, gain

    def flow(self, pose: Pose, ds: Optional[float] = None) -> torch.Tensor:
        ds = self.speed * self.dt if ds is None else ds
        cx, cy, cz, wx, wy, wz = self._rays(pose)
        below = wy < -1e-4
        t = self._camera_height(pose) / torch.where(below, -wy, torch.ones_like(wy))
        dx, dy, dz = self._displacement(pose, ds)
        # World displacement in the camera frame: inverse yaw, then inverse pitch.
        p, g = math.radians(self._pitch(pose)), math.radians(pose.gaze)
        x1, z1 = dx * math.cos(g) - dz * math.sin(g), dz * math.cos(g) + dx * math.sin(g)
        dcx, dcy, dcz = x1, dy * math.cos(p) - z1 * math.sin(p), dy * math.sin(p) + z1 * math.cos(p)
        px, py, pz = t * cx - dcx, t * cy - dcy, t * cz - dcz
        u_new = torch.rad2deg(torch.atan2(px, pz))
        v_new = torch.rad2deg(torch.atan2(py, torch.hypot(px, pz)))
        uu, vv = self._angles()
        flow = torch.stack((u_new - uu, v_new - vv)) * below.float()
        return flow

    # -- trials -----------------------------------------------------------------

    def generate_trial(self, seed: int = 0) -> Trial:
        rng = _rng(seed)
        present = bool(rng.random() < self.present_fraction)
        offset = 0.0
        if present:
            offset = float(rng.choice([-1.0, 1.0]) * rng.uniform(2 * self.delta_min, self.h_max))
        pert = dataclasses.replace(self.perturbation, mean=offset)
        heading0 = float(pert.sample(self.n_steps, seed=seed + 1)[0])
        h, w = self.size
        va_v = self.visual_angle * h / w
        col = int(round((heading0 / self.visual_angle + 0.5) * (w - 1)))
        row = int(round((0.5 - (-self.pitch) / va_v) * (h - 1)))
        return Trial(seed=seed, target_present=present, target_location=(row, col), heading_offset=offset,
                     metadata={"stimulus": self.stimulus.params})

    def _error(self, old: Pose, new: Pose, action: Action) -> float:
        if action.heading_estimate is None:
            return float("nan")
        return abs(action.heading_estimate - old.heading)


class LandingTask(HorizontalPlaneTask):
    """Landing on the ground plane with a constant angle of attack.

    The camera descends along a straight path that makes the constant angle
    ``attack_angle`` with the horizontal (90 deg is a vertical descent, which gives
    a purely ventral optic flow), without roll or pitch. Its speed along the path
    is ``speed_command + d(t)`` where ``d`` is the exogenous
    :class:`Perturbation` (m/s). The model controls ``speed_command`` and should
    touch the ground as fast and as softly as possible; the softness is measured
    by the speed at contact.

    The luminance gain depends only on the direction of the pixel, so that it does
    not change with the altitude and the brightness-constancy assumption of the
    optic-flow estimators holds.

    Args:
        stimulus: Texture, see :class:`Task`.
        size: ``(height, width)`` of the images (pixels).
        visual_angle: Visual angle of the image width (deg).
        attack_angle: Angle between the path and the ground (deg).
        altitude0: Initial altitude (m).
        speed0: Nominal speed along the path (m/s).
        z_contact: Altitude below which the camera touches the ground (m).
        v_safe: Largest contact speed of a successful landing (m/s).
        v_max: Largest speed along the path (m/s).
        max_time: Duration after which an episode is a failure (s).
        period: Period of the texture on the ground (m).
        **kwargs: Passed to :class:`HorizontalPlaneTask` (``dt``, ``noise_std``,
            ``perturbation``, ``depth_exponent``, ...).
    """

    def __init__(
        self,
        stimulus: Any,
        size: Union[int, Tuple[int, int]] = 128,
        visual_angle: float = 45.0,
        attack_angle: float = 90.0,
        altitude0: float = 10.0,
        speed0: float = 2.0,
        z_contact: float = 0.05,
        v_safe: float = 0.5,
        v_max: float = 10.0,
        max_time: float = 60.0,
        period: float = 20.0,
        **kwargs: Any,
    ) -> None:
        dt = kwargs.setdefault("dt", 1 / 30)
        kwargs.setdefault("perturbation", Perturbation(dt=dt, tau=1.0, sigma_ou=0.1, hazard_rate=0.2, sigma_jump=0.3,
                                                       jump_tau=1.0))
        kwargs.setdefault("depth_exponent", 0.0)
        kwargs.setdefault("noise_std", 0.0)
        kwargs["n_steps"] = int(round(max_time / dt))
        super().__init__(stimulus, size, visual_angle, period=period, warmup=0, **kwargs)
        self.attack_angle = float(attack_angle)
        self.altitude0 = float(altitude0)
        self.speed0 = float(speed0)
        self.z_contact = float(z_contact)
        self.v_safe = float(v_safe)
        self.v_max = float(v_max)
        self.delta_min = None  # no heading decision in the landing task

    def _camera_height(self, pose: Pose) -> float:
        return pose.altitude

    def _pitch(self, pose: Pose) -> float:
        return -self.attack_angle

    def _displacement(self, pose: Pose, ds: float) -> Tuple[float, float, float]:
        a = math.radians(self.attack_angle)
        return 0.0, -ds * math.sin(a), ds * math.cos(a)

    def generate_trial(self, seed: int = 0) -> Trial:
        rng = _rng(seed)
        z0 = self.altitude0 * (1.0 + 0.1 * rng.uniform(-1.0, 1.0))
        return Trial(seed=seed, heading_offset=0.0, metadata={"altitude0": float(z0), "stimulus": self.stimulus.params})

    def reset(self, trial: Optional[Trial] = None, seed: Optional[int] = None, closed_loop: bool = True) -> Observation:
        if trial is None:
            trial = self.generate_trial(0 if seed is None else seed)
        self._altitude0_trial = trial.metadata.get("altitude0", self.altitude0)
        self._v_cmd = self.speed0
        return super().reset(trial, seed, closed_loop)

    def _initial_pose(self, trial: Trial) -> Pose:
        return Pose(altitude=self._altitude0_trial)

    def step(self, action: Optional[Action] = None) -> Tuple[Observation, bool, Dict[str, Any]]:
        action = self._hold(action)
        if self._closed_loop and action.speed_command is not None:
            self._v_cmd = float(action.speed_command)
        old = self.pose
        speed = float(np.clip(self._v_cmd + self._d[self._k + 1], 0.0, self.v_max))
        ds = speed * self.dt
        a = math.radians(self.attack_angle)
        new = Pose(distance=old.distance + ds, altitude=old.altitude - ds * math.sin(a), z=old.z + ds * math.cos(a))
        self._k += 1
        self.pose = new
        landed = new.altitude <= self.z_contact
        done = landed or self._k >= self.n_steps
        info: Dict[str, Any] = {"error": float("nan"), "speed": speed, "altitude_obs": old.altitude,
                                "gaze_obs": 0.0, "heading_obs": 0.0, "foe_obs": 0.0}
        if done:
            info.update(
                landed=landed, time_to_land=self._k * self.dt, contact_speed=speed,
                vertical_contact_speed=speed * math.sin(a), success=bool(landed and speed <= self.v_safe),
            )
        obs = self._observe(new, self._prev)
        if self.control_due:  # the next decision compares its frame with this one
            self._prev = obs.image
        return obs, done, info


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ActiveVisionModel:
    """Base class of the models that are evaluated by a :class:`Benchmark`.

    At each step the model receives an :class:`Observation` and returns an
    :class:`Action`: a rotation of the gaze (cylinder, plane), a commanded speed
    (landing) and, for the estimation tasks, an estimate of the heading.
    Sub-class it and override :meth:`act`.
    """

    def reset(self, seed: Optional[int] = None) -> None:
        """Prepare the model for a new trial.

        Args:
            seed: Seed of the trial. Stochastic models derive their generator from
                it so that results do not depend on the order of the trials.
        """

    def act(self, observation: Observation) -> Action:
        """Return the action for the current observation."""
        raise NotImplementedError


class RandomHeadingModel(ActiveVisionModel):
    """Chance baseline: a heading drawn uniformly at random.

    Args:
        h_max: Half-width of the range of the estimates (deg).
        seed: Seed of the model.
    """

    def __init__(self, h_max: float = 10.0, seed: Optional[int] = 0) -> None:
        self.h_max = float(h_max)
        self.seed = seed
        self._rng = _rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self._rng = _rng([0 if self.seed is None else self.seed, 0 if seed is None else seed])

    def act(self, observation: Observation) -> Action:
        return Action(heading_estimate=float(self._rng.uniform(-self.h_max, self.h_max)))


class FixedGazeModel(ActiveVisionModel):
    """Baseline that never moves its gaze and does not estimate anything."""

    def act(self, observation: Observation) -> Action:
        return Action()


class FlowHeadingModel(ActiveVisionModel):
    """Estimate the heading from the optic flow of two successive frames.

    The FoE of the flow is estimated with :func:`estimate_foe`, optionally smoothed over
    time (``alpha``) and converted into an estimate of the heading in the world.

    Args:
        visual_angle: Visual angle of the image width (deg). ``None`` (default)
            reads the one of the camera in the observation; a value overrides it.
        elevation: Elevation of the FoE in the image (deg).
        alpha: Weight of the new estimate in the exponential smoothing (0-1). The default
            1 switches the smoothing off: the estimate is the one of the last frames and does
            not lag behind the target. A smaller value averages the noise of the estimate over
            about ``1 / alpha`` decisions, at the price of that lag.
        window: Size of the windows where the inverse depth is constant (pixels).
        yaw_range: Largest unknown yaw rotation per frame that is fitted jointly
            with the FoE (deg per frame), see :func:`estimate_foe`. It is 0 by
            default, since the heading of a camera over a static ground plane does
            not rotate the image.
    """

    def __init__(
        self,
        visual_angle: Optional[float] = None,
        elevation: float = 0.0,
        alpha: float = 1.0,
        window: int = 9,
        yaw_range: float = 0.0,
    ) -> None:
        self.visual_angle = None if visual_angle is None else float(visual_angle)
        self.elevation = float(elevation)
        self.alpha = float(alpha)
        self.window = int(window)
        self.yaw_range = float(yaw_range)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        self._foe = 0.0
        self._started = False
        self._gaze_before: Optional[float] = None

    def _update(self, observation: Observation) -> float:
        if observation.previous is not None:
            # The rotation of the gaze between the two frames is sensed by the
            # observer; its effect on the image is removed before the estimation.
            rotation = 0.0 if self._gaze_before is None else observation.gaze - self._gaze_before
            visual_angle = observation.visual_angle if self.visual_angle is None else self.visual_angle
            foe = estimate_foe(observation.previous, observation.image, visual_angle, self.elevation, self.window,
                               rotation=rotation, yaw_range=self.yaw_range)
            if math.isfinite(foe):
                self._foe = self.alpha * foe + (1 - self.alpha) * self._foe if self._started else foe
                self._started = True
        self._gaze_before = observation.gaze
        return self._foe

    def act(self, observation: Observation) -> Action:
        foe = self._update(observation)
        return Action(heading_estimate=observation.gaze + foe)


class GazeStabilisationModel(FlowHeadingModel):
    """Keep the gaze on the heading: rotate the gaze towards the estimated FoE.

    The yaw of the tunnel with respect to the camera adds a uniform flow, which
    is fitted jointly with the FoE (``yaw_range`` is 0.2 deg per frame by default).

    Args:
        gain: Fraction of the estimated FoE offset that is corrected at each step.
        **kwargs: Passed to :class:`FlowHeadingModel`.
    """

    def __init__(self, gain: float = 0.4, **kwargs: Any) -> None:
        kwargs.setdefault("yaw_range", 0.2)
        super().__init__(**kwargs)
        self.gain = float(gain)

    def act(self, observation: Observation) -> Action:
        foe = self._update(observation)
        return Action(gaze_shift=self.gain * foe, heading_estimate=observation.gaze + foe)


class DummyController(ActiveVisionModel):
    """Test double of an active controller with a noisy and delayed FoE estimate.

    It reads the true FoE from ``observation.info`` and returns it with a Gaussian
    noise and a delay of ``lag`` steps, then applies the same proportional gaze
    correction as :class:`GazeStabilisationModel`. It tests the closed loop (heading
    dynamics, perturbation) independently of the image-based estimation.

    A proportional correction of a delayed estimate is stable only for a small
    gain: for a delay of ``lag`` steps the loop oscillates when
    ``gain > 2 sin(pi / (4 lag + 2))`` (0.45 for 3 steps, 0.15 for 10 steps).

    Args:
        noise_std: Standard deviation of the noise of the estimate (deg).
        lag: Delay of the estimate (steps).
        gain: Fraction of the estimated FoE offset that is corrected at each step.
        seed: Seed of the noise.
    """

    def __init__(self, noise_std: float = 1.0, lag: int = 3, gain: float = 0.15, seed: Optional[int] = 0) -> None:
        self.noise_std = float(noise_std)
        self.lag = int(lag)
        self.gain = float(gain)
        self.seed = seed
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        self._rng = _rng([0 if self.seed is None else self.seed, 0 if seed is None else seed])
        self._history: List[float] = []

    def act(self, observation: Observation) -> Action:
        self._history.append(float(observation.info["foe"]))
        delayed = self._history[max(0, len(self._history) - 1 - self.lag)]
        foe = delayed + self.noise_std * float(self._rng.standard_normal())
        return Action(gaze_shift=self.gain * foe, heading_estimate=observation.gaze + foe)


class ConstantSpeedLandingModel(ActiveVisionModel):
    """Baseline for the landing: keep the nominal speed (which is perturbed).

    Args:
        speed0: Commanded speed (m/s).
    """

    def __init__(self, speed0: float = 2.0) -> None:
        self.speed0 = float(speed0)

    def act(self, observation: Observation) -> Action:
        return Action(speed_command=self.speed0)


class ConstantDivergenceLandingModel(ActiveVisionModel):
    """Land by keeping the divergence of the optic flow constant.

    The bio-inspired strategy observed in honeybees and studied by Guido de Croon
    keeps the divergence of the optic flow ``D = -v_z / z`` constant, ``D = C``,
    where ``v_z`` is the vertical velocity (positive upwards) and ``z`` the height
    above the surface (Izzo & de Croon, 2012, J. Guid. Control Dyn. 35(4)). The
    height and the vertical velocity then decay exponentially, which gives a
    smooth and safe landing.

    The divergence ``D`` is estimated from two successive frames
    (:func:`estimate_divergence`); its inverse is the time to contact
    (:func:`estimate_time_to_contact`). A texture gives no depth, but the divergence
    needs only the visual angle of the camera, which the model reads from the
    observation. The commanded speed ``v`` obeys

    .. math:: \\frac{d \\ln v}{dt} = -\\hat D + \\kappa\\,(C - \\hat D),

    where the first term is the feed-forward decay that holds ``D`` constant and
    the second one is a feedback that regulates ``D`` to its set point ``C``.
    With ``L = ln(v / z)`` this gives ``dL/dt = kappa (C - D)``, which is stable
    for ``kappa > 0``.

    Args:
        divergence: Set point ``C`` of the divergence (s^-1). A larger value
            gives a faster and harder landing.
        kappa: Dimensionless gain of the feedback.
        speed0: Initial commanded speed (m/s).
        visual_angle: Visual angle of the image width (deg). ``None`` (default)
            reads the one of the camera in the observation; a value overrides it.
        alpha: Weight of the new estimate in the exponential smoothing (0-1).
        v_max: Largest commanded speed (m/s).
    """

    def __init__(
        self,
        divergence: float = 0.5,
        kappa: float = 1.0,
        speed0: float = 2.0,
        visual_angle: Optional[float] = None,
        alpha: float = 0.3,
        v_max: float = 10.0,
    ) -> None:
        self.divergence = float(divergence)
        self.kappa = float(kappa)
        self.speed0 = float(speed0)
        self.visual_angle = None if visual_angle is None else float(visual_angle)
        self.alpha = float(alpha)
        self.v_max = float(v_max)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        self._v = self.speed0
        self._d_hat: Optional[float] = None

    def act(self, observation: Observation) -> Action:
        if observation.previous is not None:
            visual_angle = observation.visual_angle if self.visual_angle is None else self.visual_angle
            d = estimate_divergence(observation.previous, observation.image, visual_angle, observation.dt)
            if math.isfinite(d):
                d = max(d, 0.0)
                self._d_hat = d if self._d_hat is None else self.alpha * d + (1 - self.alpha) * self._d_hat
        if self._d_hat is not None:
            rate = -self._d_hat + self.kappa * (self.divergence - self._d_hat)
            self._v = float(np.clip(self._v * math.exp(rate * observation.dt), 1e-3, self.v_max))
        return Action(speed_command=self._v)


def default_models(task: Task) -> Dict[str, ActiveVisionModel]:
    """Return the reference models that are bundled for a task.

    Args:
        task: A :class:`HorizontalPlaneTask`, :class:`CylinderTask` or
            :class:`LandingTask`.
    """
    if isinstance(task, LandingTask):
        return {"constant speed": ConstantSpeedLandingModel(task.speed0),
                "constant divergence": ConstantDivergenceLandingModel(speed0=task.speed0, visual_angle=task.visual_angle,
                                                                       v_max=task.v_max)}
    if isinstance(task, CylinderTask):
        return {"fixed gaze": FixedGazeModel(), "gaze stabilisation": GazeStabilisationModel(visual_angle=task.visual_angle)}
    h_max = getattr(task, "h_max", 10.0)
    return {"random": RandomHeadingModel(h_max), "flow": FlowHeadingModel(visual_angle=task.visual_angle,
                                                                          elevation=-getattr(task, "pitch", 0.0))}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Result of one model on one trial.

    Attributes:
        trial: The trial.
        estimates: Heading estimate at each step (deg, ``nan`` if none).
        truth: True heading at each step (deg).
        errors: Task error at each step (deg, ``nan`` if undefined).
        decision: Whether the model decided *heading decentred* (plane task), or
            ``None``.
        correct: Whether that decision is right, or ``None``.
        elapsed_s: Wall-clock time of the trial (s).
        info: Outcome of the last step (landing: ``success``, ``time_to_land``,
            ``contact_speed``, ...).
        warmup: Number of initial steps ignored by the error metrics.
    """

    trial: Trial
    estimates: np.ndarray
    truth: np.ndarray
    errors: np.ndarray
    decision: Optional[bool]
    correct: Optional[bool]
    elapsed_s: float
    info: Dict[str, Any] = field(default_factory=dict)
    warmup: int = 0

    @property
    def error(self) -> float:
        """Mean absolute error over the trial (deg), ``nan`` if undefined."""
        e = self.errors[self.warmup:]
        return float(np.nanmean(e)) if np.any(np.isfinite(e)) else float("nan")

    @property
    def mean_square_error(self) -> float:
        """Mean squared error over the trial (deg^2), ``nan`` if undefined."""
        e = self.errors[self.warmup:]
        return float(np.nanmean(e**2)) if np.any(np.isfinite(e)) else float("nan")


@dataclass
class BenchmarkResult:
    """Aggregated results of a model on a set of trials.

    The estimation tasks report the accuracy of the estimate (``rms_error``,
    ``median_error``) and the signal-detection metrics of Table 1 of the paper
    (``accuracy``, ``hit_rate``, ``false_alarm_rate``, ``d_prime``); the landing
    reports ``success_rate``, ``mean_time_to_land`` and ``mean_contact_speed``.

    Attributes:
        trial_results: Per-trial :class:`TrialResult` objects.
        model_name: Name of the evaluated model.
    """

    trial_results: List[TrialResult]
    model_name: str = "unknown"

    # -- error of the estimate or of the control -----------------------------

    @property
    def rms_error(self) -> float:
        """Root-mean-square error over all the trials (deg)."""
        v = [r.mean_square_error for r in self.trial_results]
        return float(np.sqrt(np.nanmean(v))) if np.any(np.isfinite(v)) else float("nan")

    @property
    def median_error(self) -> float:
        """Median over the trials of the mean absolute error (deg)."""
        v = [r.error for r in self.trial_results]
        return float(np.nanmedian(v)) if np.any(np.isfinite(v)) else float("nan")

    # -- signal detection (Table 1 of the paper) -------------------------------

    def _decided(self) -> List[TrialResult]:
        return [r for r in self.trial_results if r.decision is not None]

    @property
    def accuracy(self) -> float:
        """``(hits + correct rejections) / N``."""
        rs = self._decided()
        return sum(bool(r.correct) for r in rs) / len(rs) if rs else float("nan")

    def hit_rate(self) -> float:
        """``hits / N_+``: decisions *decentred* among the target-present trials."""
        present = [r for r in self._decided() if r.trial.target_present]
        return sum(bool(r.decision) for r in present) / len(present) if present else float("nan")

    def false_alarm_rate(self) -> float:
        """``false alarms / N_-``: decisions *decentred* among the target-absent trials."""
        absent = [r for r in self._decided() if not r.trial.target_present]
        return sum(bool(r.decision) for r in absent) / len(absent) if absent else float("nan")

    def d_prime(self) -> float:
        """Sensitivity ``z(H) - z(F)``, with the loglinear correction of Hautus (1995).

        The correction adds 0.5 to the counts and 1 to the number of trials, so
        that the result is finite when a rate is 0 or 1.
        """
        rs = self._decided()
        present = [r for r in rs if r.trial.target_present]
        absent = [r for r in rs if not r.trial.target_present]
        if not present or not absent:
            return float("nan")
        hits, false_alarms = sum(bool(r.decision) for r in present), sum(bool(r.decision) for r in absent)
        z = statistics.NormalDist().inv_cdf
        return float(z((hits + 0.5) / (len(present) + 1)) - z((false_alarms + 0.5) / (len(absent) + 1)))

    # -- landing -------------------------------------------------------------

    def _landed(self) -> List[TrialResult]:
        return [r for r in self.trial_results if "success" in r.info]

    @property
    def success_rate(self) -> float:
        """Fraction of the landings that touched the ground at a speed below ``v_safe``."""
        rs = self._landed()
        return sum(bool(r.info["success"]) for r in rs) / len(rs) if rs else float("nan")

    @property
    def mean_time_to_land(self) -> float:
        """Mean duration of the episodes (s)."""
        rs = self._landed()
        return float(np.mean([r.info["time_to_land"] for r in rs])) if rs else float("nan")

    @property
    def mean_contact_speed(self) -> float:
        """Mean speed at contact (m/s)."""
        rs = self._landed()
        return float(np.mean([r.info["contact_speed"] for r in rs])) if rs else float("nan")

    @property
    def mean_elapsed_s(self) -> float:
        """Mean wall-clock time per trial (s)."""
        return float(np.mean([r.elapsed_s for r in self.trial_results])) if self.trial_results else float("nan")

    def summary(self) -> str:
        """Return a human-readable summary of the metrics that apply."""
        lines = [f"Model            : {self.model_name}", f"Trials           : {len(self.trial_results)}"]
        if np.isfinite(self.rms_error):
            lines += [f"RMS error (deg)  : {self.rms_error:.3f}", f"Median error (deg): {self.median_error:.3f}"]
        if self._decided():
            lines += [f"Accuracy         : {self.accuracy:.3f}", f"Hit rate         : {self.hit_rate():.3f}",
                      f"False alarm rate : {self.false_alarm_rate():.3f}", f"d'               : {self.d_prime():.3f}"]
        if self._landed():
            lines += [f"Success rate     : {self.success_rate:.3f}", f"Time to land (s) : {self.mean_time_to_land:.2f}",
                      f"Contact speed (m/s): {self.mean_contact_speed:.3f}"]
        lines.append(f"Mean time (s)    : {self.mean_elapsed_s:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark layer
# ---------------------------------------------------------------------------


def _run_trial_job(args: Tuple[Any, ...]) -> TrialResult:
    """Run one trial in a child process (the child always computes on the CPU)."""
    task, model, trial, closed_loop = args
    torch.set_num_threads(1)
    task.to("cpu")
    return Benchmark(task, closed_loop=closed_loop).run_trial(model, trial)


class Benchmark:
    """Evaluate models on a :class:`Task`.

    This is the entry point of the paper: a benchmark generates a dataset of
    trials from the task (:meth:`generate_dataset`) and evaluates models on it
    (:meth:`evaluate`). It plays every trial as an episode: the model receives an
    observation, returns an action, and the task moves the camera. See
    :class:`PassiveBenchmark` (the actions are ignored) and
    :class:`ActiveBenchmark` (the actions control the camera).

    Args:
        task: The task to probe.
        models: Mapping from names to models. Defaults to :func:`default_models`.
        closed_loop: If ``True``, the actions of the models are applied.

    Examples:
        >>> task = HorizontalPlaneTask(GridStimulus(), size=32, n_steps=6)
        >>> bench = Benchmark(task)
        >>> trials = bench.generate_dataset(n_trials=2, seed=0)
        >>> sorted(bench.evaluate(trials, n_jobs=1))
        ['flow', 'random']
    """

    def __init__(
        self,
        task: Optional[Task] = None,
        models: Optional[Dict[str, ActiveVisionModel]] = None,
        closed_loop: bool = True,
    ) -> None:
        self.task = task
        self.models = models
        self.closed_loop = closed_loop

    def generate_dataset(self, n_trials: int = 100, seed: Optional[int] = None) -> List[Trial]:
        """Generate ``n_trials`` reproducible trials with the task.

        Args:
            n_trials: Number of trials.
            seed: Master seed. Defaults to the seed of the task.
        """
        return self.task.generate_dataset(n_trials, seed)

    def run_trial(self, model: ActiveVisionModel, trial: Trial) -> TrialResult:
        """Play one episode of *model* on *trial*.

        The model decides every ``task.control_every`` camera steps; between two
        decisions the last action is held, so that the estimates, the truth and the
        errors have one value per camera step.

        Args:
            model: The model.
            trial: The trial.
        """
        t0 = time.perf_counter()
        task = self.task
        obs = task.reset(trial, closed_loop=self.closed_loop)
        model.reset(trial.seed)
        estimates, truth, errors = [], [], []
        info: Dict[str, Any] = {}
        done = False
        while not done:
            if task.control_due:
                action = model.act(obs)
            obs, done, info = task.step(action)
            estimates.append(np.nan if action.heading_estimate is None else action.heading_estimate)
            truth.append(info["heading_obs"])
            errors.append(info["error"])
        decision = correct = None
        delta_min = getattr(task, "delta_min", None)
        if delta_min is not None and np.isfinite(estimates[-1]):
            decision = bool(abs(estimates[-1] - info["gaze_obs"]) > delta_min)
            correct = decision == trial.target_present
        return TrialResult(
            trial=trial, estimates=np.asarray(estimates), truth=np.asarray(truth), errors=np.asarray(errors),
            decision=decision, correct=correct, elapsed_s=time.perf_counter() - t0,
            info={k: v for k, v in info.items() if k in ("landed", "success", "time_to_land", "contact_speed",
                                                          "vertical_contact_speed")},
            warmup=task.warmup,
        )

    def run(
        self,
        model: ActiveVisionModel,
        trials: Sequence[Trial],
        model_name: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ) -> BenchmarkResult:
        """Evaluate *model* on *trials*.

        Args:
            model: The model.
            trials: The trials.
            model_name: Display name. Defaults to the class name of the model.
            n_jobs: Number of processes. Defaults to all the CPU cores for 48 trials
                or more, and to a single process for fewer trials (starting the
                processes costs a few seconds).
        """
        name = model_name if model_name is not None else type(model).__name__
        if n_jobs is None:
            n_jobs = (os.cpu_count() or 1) if len(trials) >= 48 else 1
        if n_jobs > 1:
            jobs = [(self.task, model, t, self.closed_loop) for t in trials]
            results = parallel_map(_run_trial_job, jobs, n_jobs)
        else:
            results = [self.run_trial(model, t) for t in trials]
        return BenchmarkResult(trial_results=results, model_name=name)

    def compare(
        self,
        models: Dict[str, ActiveVisionModel],
        trials: Sequence[Trial],
        n_jobs: Optional[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Evaluate several models on the same trials.

        Args:
            models: Mapping from names to models.
            trials: The shared trials.
            n_jobs: Number of processes, see :meth:`run`.
        """
        return {name: self.run(m, trials, model_name=name, n_jobs=n_jobs) for name, m in models.items()}

    def evaluate(
        self,
        trials: Sequence[Trial],
        models: Optional[Dict[str, ActiveVisionModel]] = None,
        n_jobs: Optional[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Evaluate the models of the benchmark (or the bundled ones) on *trials*.

        Args:
            trials: The trials, for instance from :meth:`generate_dataset`.
            models: Mapping from names to models. Defaults to the models given to
                the benchmark, then to :func:`default_models`.
            n_jobs: Number of processes, see :meth:`run`.

        Returns:
            One :class:`BenchmarkResult` per model.
        """
        models = models or self.models or default_models(self.task)
        return self.compare(models, trials, n_jobs=n_jobs)


class ActiveBenchmark(Benchmark):
    """Closed-loop benchmark: the actions of the model control the camera.

    The gaze rotates with ``Action.gaze_shift`` and the heading follows the gaze
    (``Task.heading_lag``); in the landing task ``Action.speed_command`` sets the
    speed.
    """

    def __init__(self, task: Optional[Task] = None, models: Optional[Dict[str, ActiveVisionModel]] = None) -> None:
        super().__init__(task, models, closed_loop=True)

    def run_episode(self, model: ActiveVisionModel, trial: Trial) -> TrialResult:
        """Play one closed-loop episode (an alias of :meth:`run_trial`)."""
        return self.run_trial(model, trial)


class PassiveBenchmark(Benchmark):
    """Open-loop benchmark: the camera follows its own course and generates streams of images.

    The actions of the models are ignored (the observer only watches), and
    :meth:`stream` yields the frames with their ground-truth optic flow.
    """

    def __init__(self, task: Optional[Task] = None, models: Optional[Dict[str, ActiveVisionModel]] = None) -> None:
        super().__init__(task, models, closed_loop=False)

    def stream(
        self,
        n_frames: int,
        seed: int = 0,
        trial: Optional[Trial] = None,
        perturbation: Optional[Perturbation] = None,
        with_flow: bool = True,
        colour: bool = True,
    ) -> Iterator[Frame]:
        """Generate a stream of frames.

        Args:
            n_frames: Number of frames (at most ``task.n_steps + 1``).
            seed: Seed of the stream, if no trial is given.
            trial: A trial to play instead.
            perturbation: Perturbation of the heading that replaces the one of the
                task, for instance with jumps.
            with_flow: If ``True``, also compute the ground-truth optic flow.
            colour: If ``True`` and the texture has colours, yield RGB images.

        Yields:
            :class:`Frame` objects.
        """
        task = copy.copy(self.task)
        if perturbation is not None:
            task.perturbation = perturbation
        task.n_steps = max(task.n_steps, n_frames)
        task.reset(trial, seed=seed, closed_loop=False)
        for k in range(n_frames):
            pose = dataclasses.replace(task.pose)
            img = task.render(pose).cpu().numpy()
            img = img[0] if img.shape[0] == 1 else (np.moveaxis(img, 0, -1) if colour else img.mean(axis=0))
            flow = task.flow(pose).cpu().numpy() if with_flow else None
            yield Frame(image=img, pose=pose, flow=flow, t=k * task.dt, perturbation=float(task._d[k]))
            task.step(Action())

    @staticmethod
    def to_video(frames: Iterable[Frame], path: str, fps: int = 60, crf: int = 28) -> str:
        """Write frames to a video file (requires ``imageio`` with ``ffmpeg``).

        Args:
            frames: Frames, for instance from :meth:`stream`.
            path: Output path (``.mp4``).
            fps: Frames per second.
            crf: Constant rate factor of the H.264 encoder (0-51): the larger, the
                smaller the file and the lower the quality.
        """
        return _write_video((frame.image for frame in frames), path, fps, crf)


# ---------------------------------------------------------------------------
# Videos (optional: the encoding needs imageio with ffmpeg, the overlays need Pillow)
# ---------------------------------------------------------------------------

#: Colours (RGB, 0-1) of the overlays of the videos: the true target (a dot) and its
#: estimate (a circle). They come from the colour-blind safe Okabe-Ito palette.
TARGET_COLOUR: Tuple[float, float, float] = (0.835, 0.369, 0.0)
ESTIMATE_COLOUR: Tuple[float, float, float] = (0.337, 0.706, 0.914)


def to_rgb(image: Any) -> np.ndarray:
    """Return an image as a ``(H, W, 3)`` float array in [0, 1].

    Args:
        image: A ``(H, W)`` or ``(H, W, 3)`` array, or a ``(C, H, W)`` tensor (the
            layout of :meth:`Task.render`), with floats in [0, 1] or ``uint8``.
    """
    if isinstance(image, torch.Tensor):
        image = np.moveaxis(image.detach().cpu().numpy(), 0, -1)
    image = np.asarray(image)
    if image.dtype == np.uint8:
        image = image / 255.0
    image = np.asarray(image, dtype=float)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    return np.repeat(image[:, :, None], 3, axis=2) if image.ndim == 2 else image


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.rint(255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)


def draw_marker(
    image: np.ndarray,
    row: float,
    col: float,
    radius: float = 4.0,
    colour: Tuple[float, float, float] = TARGET_COLOUR,
    ring: bool = False,
    width: float = 1.5,
) -> np.ndarray:
    """Draw a dot or a circle on an image, anti-aliased and on a dark halo.

    The halo keeps the marker visible on any texture, bright or dark.

    Args:
        image: ``(H, W, 3)`` image in [0, 1].
        row: Row of the centre (pixels, downwards).
        col: Column of the centre (pixels, to the right).
        radius: Radius of the dot, or outer radius of the circle (pixels).
        colour: RGB colour in [0, 1].
        ring: If ``True``, draw a circle instead of a dot.
        width: Thickness of the circle (pixels).

    Returns:
        A new ``(H, W, 3)`` image. A centre that is not finite leaves it unchanged.

    Examples:
        >>> out = draw_marker(np.zeros((9, 9, 3)), 4, 4, radius=2.0, colour=(1.0, 1.0, 1.0))
        >>> float(out[4, 4, 0]), float(out[0, 0, 0])
        (1.0, 0.0)
    """
    out = np.array(image, dtype=float)
    if not (math.isfinite(row) and math.isfinite(col)):
        return out
    rows, cols = np.indices(out.shape[:2])
    distance = np.hypot(rows - row, cols - col)
    inner = radius - width if ring else -1.0

    def paint(rgb: Sequence[float], opacity: float, grow: float) -> None:
        edge = np.minimum(distance - (inner - grow), (radius + grow) - distance)
        alpha = (np.clip(edge + 0.5, 0.0, 1.0) * opacity)[:, :, None]
        out[:] = out * (1 - alpha) + np.asarray(rgb) * alpha

    paint((0.0, 0.0, 0.0), 0.6, 1.5)
    paint(colour, 1.0, 0.0)
    return out


def draw_text(
    image: np.ndarray,
    text: str,
    position: Tuple[float, float],
    anchor: str = "lt",
    colour: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    size: int = 10,
) -> np.ndarray:
    """Write a text on an image, with a dark outline (needs Pillow, which torchvision requires).

    Args:
        image: ``(H, W, 3)`` image in [0, 1].
        text: The text.
        position: ``(x, y)`` of the corner of the text that ``anchor`` names (pixels).
        anchor: Two letters: ``l`` or ``r`` (left or right edge of the text) then ``t``
            or ``b`` (top or bottom edge).
        colour: RGB colour in [0, 1].
        size: Height of the font (pixels).

    Returns:
        A new ``(H, W, 3)`` image.
    """
    from PIL import Image, ImageDraw, ImageFont  # lazy: only the videos need it

    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 only has a bitmap font of a fixed size
        font = ImageFont.load_default()
    canvas = Image.fromarray(_to_uint8(image))
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    x = position[0] - (right if anchor[0] == "r" else left)
    y = position[1] - (bottom if anchor[1] == "b" else top)
    draw.text((x, y), text, fill=tuple(int(255 * c) for c in colour), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(canvas, dtype=float) / 255.0


def tile(images: Sequence[np.ndarray], ncols: Optional[int] = None, gap: int = 2, fill: float = 0.0) -> np.ndarray:
    """Arrange images of the same size in a grid, row by row.

    Args:
        images: ``(H, W, 3)`` images.
        ncols: Number of columns. Defaults to one row.
        gap: Space between the images (pixels).
        fill: Value of the gap and of the empty cells.

    Examples:
        >>> a = np.zeros((4, 6, 3))
        >>> tile([a, a, a], ncols=2, gap=2).shape
        (10, 14, 3)
    """
    ncols = len(images) if ncols is None else ncols
    nrows = -(-len(images) // ncols)
    h, w = images[0].shape[:2]
    canvas = np.full((nrows * h + (nrows - 1) * gap, ncols * w + (ncols - 1) * gap, 3), fill, dtype=images[0].dtype)
    for k, image in enumerate(images):
        r, c = divmod(k, ncols)
        canvas[r * (h + gap): r * (h + gap) + h, c * (w + gap): c * (w + gap) + w] = image
    return canvas


def _write_video(frames: Iterable[Any], path: str, fps: int, crf: int) -> str:
    """Encode frames in H.264 (``yuv420p``, which every browser plays); see :func:`save_video`."""
    import imageio.v2 as imageio  # lazy: imageio is optional

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # macro_block_size=1 keeps the size of the frames; the odd ones are padded below, since yuv420p needs even sizes.
    options = dict(fps=fps, codec="libx264", pixelformat="yuv420p", quality=None, macro_block_size=1,
                   output_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"])
    with imageio.get_writer(path, **options) as writer:
        for frame in frames:
            img = _to_uint8(to_rgb(frame))
            img = np.pad(img, ((0, img.shape[0] % 2), (0, img.shape[1] % 2), (0, 0)), mode="edge")
            writer.append_data(img)
    return path


def save_video(
    frames: Iterable[Any],
    name: str,
    directory: str = "figures",
    fps: int = 30,
    crf: int = 28,
    force: bool = False,
) -> str:
    """Write frames to ``{directory}/{name}.mp4`` (requires ``imageio`` with ``ffmpeg``).

    The default ``crf`` gives light files (a few hundred kilobytes for a few seconds of
    a ``128 x 128`` image), small enough to be embedded in a notebook with :func:`show_video`.

    Like :func:`save_figure`, it writes in ``directory`` only if the switch
    :data:`SAVE_FIGURES` is on (``AVB_FIGURES=1``). Otherwise the video goes to a
    temporary directory: a notebook can still embed it with :func:`show_video`, and the
    tracked files of ``figures/`` stay untouched.

    Args:
        frames: Images, see :func:`to_rgb`.
        name: File name without extension.
        directory: Output directory (created if needed).
        fps: Frames per second.
        crf: Constant rate factor of the H.264 encoder (0-51): the larger, the
            smaller the file and the lower the quality.
        force: If ``True``, write in ``directory`` whatever the switch.

    Returns:
        The path of the video.
    """
    if not (SAVE_FIGURES or force):
        directory = os.path.join(tempfile.gettempdir(), "avb_videos")
    return _write_video(frames, os.path.join(directory, f"{name}.mp4"), fps, crf)


def show_video(path: str, width: Union[int, str] = "100%", autoplay: bool = True) -> Any:
    """Display a video in a notebook, **embedded** in the output.

    The file is stored in the notebook (base64), so the video is shown by Jupyter, VS Code
    and by the documentation (``nbsphinx``) alike, without a link to a file that may not
    be found. It is silent, so that browsers allow it to play by themselves.

    Args:
        path: Path of the ``.mp4`` file, for instance from :func:`save_video`.
        width: Width of the player: a CSS length such as ``"100%"`` (the default: the video
            fills the width of the page and keeps its proportions) or a number of pixels.
        autoplay: If ``True``, the video starts by itself and loops.

    Returns:
        An ``IPython.display.Video``, which is displayed when it ends a cell.
    """
    from IPython.display import Video  # lazy: IPython is optional

    attributes = "controls loop muted playsinline" + (" autoplay" if autoplay else "")
    if isinstance(width, str):  # a CSS length, applied by the style so that the height follows
        return Video(os.fspath(path), embed=True, mimetype="video/mp4",
                     html_attributes=f'{attributes} style="width: {width}; height: auto;"')
    return Video(os.fspath(path), embed=True, mimetype="video/mp4", width=width, html_attributes=attributes)


def _foe_pixel(task: Task, azimuth: float, elevation: float = 0.0) -> Tuple[float, float]:
    """``(row, col)`` of a direction ``(azimuth, elevation)`` (deg) of the image of *task*."""
    h, w = task.size
    return (0.5 - elevation / (task.visual_angle * h / w)) * (h - 1), (azimuth / task.visual_angle + 0.5) * (w - 1)


def _foe_estimator(task: Task) -> ActiveVisionModel:
    """Bundled model that estimates the FoE of *task* from its images (a passive observer)."""
    return FlowHeadingModel(visual_angle=task.visual_angle, elevation=-getattr(task, "pitch", 0.0),
                            yaw_range=0.2 if isinstance(task, CylinderTask) else 0.0)


def _annotate(
    task: Task,
    image: np.ndarray,
    title: str,
    t: float,
    foe: Optional[float] = None,
    estimate: Optional[float] = None,
    elevation: float = 0.0,
    left: str = "",
    right: str = "",
) -> np.ndarray:
    """Add to a frame its title, its time, the true FoE (dot), its estimate (circle) and two corner texts."""
    h, w = image.shape[:2]
    scale, margin, size = w / 192, 3, max(8, round(w / 15))
    if foe is not None:
        image = draw_marker(image, *_foe_pixel(task, foe, elevation), radius=max(2.5, 3 * scale), colour=TARGET_COLOUR)
    if estimate is not None:
        image = draw_marker(image, *_foe_pixel(task, estimate, elevation), radius=max(5.0, 7 * scale),
                            colour=ESTIMATE_COLOUR, ring=True)
    image = draw_text(image, title, (margin, margin), "lt", size=size)
    image = draw_text(image, f"{t:.1f} s", (w - margin, margin), "rt", size=size)
    if left:
        image = draw_text(image, left, (margin, h - margin), "lb", size=size)
    if right:
        image = draw_text(image, right, (w - margin, h - margin), "rb", size=size)
    return image


def _legend(width: int, height: int = 18) -> np.ndarray:
    """Strip that explains the dot and the circle of the overlays."""
    strip = np.zeros((height, width, 3))
    left = max(4, (width - 200) // 2)
    strip = draw_marker(strip, height / 2, left + 5, radius=3.0, colour=TARGET_COLOUR)
    strip = draw_text(strip, "true FoE", (left + 14, height / 2), "lt", size=11)
    strip = draw_marker(strip, height / 2, left + 95, radius=5.5, colour=ESTIMATE_COLOUR, ring=True)
    return draw_text(strip, "estimated FoE", (left + 105, height / 2), "lt", size=11)


def _per_panel(value: Any, label: str) -> Any:
    """Return ``value[label]`` if *value* is a dictionary, and *value* itself otherwise."""
    return value.get(label) if isinstance(value, dict) else value


def passive_video(
    tasks: Dict[str, Task],
    name: str,
    directory: str = "figures",
    duration: float = 5.0,
    seed: int = 0,
    heading_offset: Union[None, float, Dict[str, float]] = None,
    show_foe: bool = False,
    models: Optional[Dict[str, ActiveVisionModel]] = None,
    ncols: Optional[int] = None,
    fps: int = 30,
    crf: int = 28,
) -> str:
    """Write the video of passive streams, one panel per task, side by side.

    The camera follows its own course (open loop). It is the way to *see* a stimulus
    or a geometry in motion. With ``show_foe`` each panel also shows the **target** of
    the navigation tasks: a dot at the true focus of expansion (FoE) and a circle at
    the FoE that a model estimates from the images (the circle appears at the second
    frame, the first one being needed to see a motion).

    The video plays in real time: with the time step ``dt`` of the tasks, one frame out
    of ``round(1 / (dt * fps))`` is kept.

    Args:
        tasks: Panels, keyed by their title. The tasks must have the same ``dt`` and
            the same image size.
        name: Name of the video (without extension).
        directory: Output directory (created if needed).
        duration: Duration of the video (s).
        seed: Seed of the trials: each task plays ``task.generate_trial(seed)``.
        heading_offset: Mean heading minus the gaze (deg) that replaces the one of the
            trial: a number for every panel, or a dictionary keyed like ``tasks``.
        show_foe: If ``True``, draw the true FoE and its estimate.
        models: Models that estimate the FoE, keyed like ``tasks`` (by default,
            :class:`FlowHeadingModel`, with the fit of the yaw for a tunnel). Only the
            ``heading_estimate`` of their action is used.
        ncols: Number of columns of the grid of panels. Defaults to one row.
        fps: Frames per second of the video.
        crf: Constant rate factor of the encoder, see :func:`save_video`.

    Returns:
        The path of the video.
    """
    if len({(task.dt, task.size) for task in tasks.values()}) != 1:
        raise ValueError("the tasks of a video must have the same time step and the same image size")
    first = next(iter(tasks.values()))
    stride = max(1, round(1 / (first.dt * fps)))
    n_steps = int(round(duration / first.dt))
    panels: List[List[np.ndarray]] = []
    for label, task in tasks.items():
        task = copy.copy(task)  # the original keeps its number of steps
        task.n_steps = max(task.n_steps, n_steps)
        trial = task.generate_trial(seed)
        offset = _per_panel(heading_offset, label)
        if offset is not None:
            trial = dataclasses.replace(trial, heading_offset=float(offset))
        model = None
        if show_foe:
            model = _per_panel(models, label) or _foe_estimator(task)
            model.reset(trial.seed)
        obs = task.reset(trial, closed_loop=False)
        elevation, estimate, frames = -getattr(task, "pitch", 0.0), None, []
        for k in range(n_steps):
            if model is not None and task.control_due:
                heading = model.act(obs).heading_estimate
                estimate = None if heading is None else heading - obs.gaze
            if k % stride == 0:
                image = _annotate(task, to_rgb(task.render(task.pose)), label, obs.t,
                                  foe=task.pose.foe if show_foe else None,
                                  estimate=estimate if obs.previous is not None else None, elevation=elevation)
                frames.append(_to_uint8(image))
            obs, _, _ = task.step()
        panels.append(frames)
    def grid() -> Iterator[np.ndarray]:
        legend = None
        for images in zip(*panels):
            image = tile(list(images), ncols)
            if show_foe and legend is None:
                legend = _to_uint8(_legend(image.shape[1]))
            yield image if legend is None else np.vstack([image, legend])

    return save_video(grid(), name, directory, fps=fps, crf=crf)


def landing_video(
    task: LandingTask,
    models: Dict[str, ActiveVisionModel],
    name: str,
    directory: str = "figures",
    seed: int = 0,
    hold: float = 1.0,
    fps: int = 30,
    crf: int = 28,
) -> str:
    """Write the video that compares landing models on the same trial, side by side.

    Each panel is the ventral view of one model. As in :func:`passive_video`, a dot marks
    the true FoE, which is the nadir of a vertical descent, and a circle marks the FoE that
    an observer estimates from the images with :class:`FlowHeadingModel`: it shows how the
    estimate degrades close to the ground. The height of the camera is written at the
    bottom right. An episode that is over keeps its last frame, with the speed at contact,
    until the last one ends (a model lands faster than another).

    Args:
        task: The landing task.
        models: Landing models, keyed by the title of their panel.
        name: Name of the video (without extension).
        directory: Output directory (created if needed).
        seed: Seed of the trial, which every model plays.
        hold: Duration of the last frame once every model has landed (s).
        fps: Frames per second of the video.
        crf: Constant rate factor of the encoder, see :func:`save_video`.

    Returns:
        The path of the video.
    """
    stride = max(1, round(1 / (task.dt * fps)))
    trial = task.generate_trial(seed)
    runs: Dict[str, Tuple[List[Tuple[np.ndarray, float, float, Optional[float]]], Dict[str, Any]]] = {}
    for label, model in models.items():
        observer = _foe_estimator(task)  # looks at the same images as the model, and does not act
        observer.reset(trial.seed)
        model.reset(trial.seed)
        obs, done, info = task.reset(trial), False, {}
        records = []
        while not done:
            heading = observer.act(obs).heading_estimate
            if obs.step % stride == 0:
                records.append((_to_uint8(to_rgb(task.render(task.pose))), obs.t, obs.info["altitude"],
                                heading - obs.gaze if obs.previous is not None else None))
            if task.control_due:
                action = model.act(obs)
            obs, done, info = task.step(action)
        runs[label] = (records, info)

    def grid() -> Iterator[np.ndarray]:
        for k in range(max(len(records) for records, _ in runs.values()) + int(round(hold * fps))):
            panels = []
            for label, (records, info) in runs.items():
                image, t, altitude, estimate = records[min(k, len(records) - 1)]
                over = k >= len(records)
                left = ""
                if over:
                    left = f"contact {info['contact_speed']:.2f} m/s" if info.get("landed") else "no contact"
                panels.append(_annotate(task, to_rgb(image), label, t, foe=0.0, estimate=estimate, left=left,
                                        right=f"z = {max(altitude, 0.0):.2f} m"))
            image = tile(panels)
            yield np.vstack([image, _legend(image.shape[1])])

    return save_video(grid(), name, directory, fps=fps, crf=crf)


# ---------------------------------------------------------------------------
# Visualisation helper (optional: requires matplotlib)
# ---------------------------------------------------------------------------


def plot_roc(results: Dict[str, BenchmarkResult], ax: Any = None) -> Any:
    """Plot the hit rate against the false-alarm rate of several models.

    Args:
        results: Mapping from model names to results (see :meth:`Benchmark.evaluate`).
        ax: Matplotlib axes. Created if ``None``.

    Returns:
        The matplotlib axes.
    """
    if ax is None:
        _, ax = new_figure()
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    for name, res in results.items():
        ax.plot(res.false_alarm_rate(), res.hit_rate(), "o", markersize=6, label=name)
    ax.set_xlabel("false alarm rate")
    ax.set_ylabel("hit rate")
    ax.set_xlim(-0.03, 1.03)  # a margin keeps the points at 0 and 1 visible
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    return ax
