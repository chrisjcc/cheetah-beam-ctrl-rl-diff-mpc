from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from matplotlib.patches import Ellipse
from scipy.ndimage import uniform_filter

plt.style.use(["science", "nature", "no-latex"])


def screen_extent(
    resolution: tuple[int, int], pixel_size: tuple[float, float]
) -> tuple[float, float, float, float]:
    """Compute extent of a diagnostic screen for Matplotlib plotting."""
    return (
        -resolution[0] * pixel_size[0] / 2,
        resolution[0] * pixel_size[0] / 2,
        -resolution[1] * pixel_size[1] / 2,
        resolution[1] * pixel_size[1] / 2,
    )


def parse_problem_index(path: Path) -> int:
    """
    Take a `path` to an episode recording according to a problems file and parse the
    problem index for it.

    Assumes one of these setups:
     - The recording is in a directory `*problem_[index]`.
     - The recording is in a `trial-[index]_*` directory.
     - The recording is in a directory `recorded_episodes` and the parent directory of
       that is `trial-[index]_*`.
    """
    if "problem_" in path.parent.name:
        return int(path.parent.name.split("_")[-1])
    elif "trial-" in path.parent.name:
        return int(path.parent.name.split("_")[0].split("-")[1])
    elif "trial_" in path.parent.name:
        return int(path.parent.name.split("_")[-1])
    else:
        return int(path.parent.parent.name.split("_")[0].split("-")[1])


def plot_screen_image(
    image: np.ndarray,
    resolution: np.ndarray,
    pixel_size: np.ndarray,
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: tuple[float, float] = (12, 10),
    save_path: Optional[Path] = None,
    vmax: Optional[float] = None,
    xlabel: bool = True,
    ylabel: bool = True,
    title: Optional[str] = None,
) -> matplotlib.axes.Axes:
    """Plot an image of a diagnostic screen."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    image_float = image.astype("float")
    image_filtered = uniform_filter(image_float, size=10)
    image_nan = image_filtered.copy()
    image_nan[image_filtered < 10] = np.nan

    ax.imshow(
        image_nan,
        vmin=0,
        vmax=image_filtered.max() if vmax is None else vmax,
        aspect="equal",
        interpolation="none",
        extent=screen_extent(resolution, pixel_size * 1e3),
        cmap="rainbow",
    )

    ax.set_title(title)

    if xlabel:
        ax.set_xlabel("x (mm)")
    if ylabel:
        ax.set_ylabel("y (mm)")

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        fig.savefig(save_path)

    return ax


def plot_beam_parameters_on_screen(
    mu_x: float,
    sigma_x: float,
    mu_y: float,
    sigma_y: float,
    resolution: Optional[np.ndarray] = None,
    pixel_size: Optional[np.ndarray] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: tuple[float, float] = (12, 10),
    save_path: Optional[Path] = None,
    xlabel: bool = True,
    ylabel: bool = True,
    title: Optional[str] = None,
    measurement_accuracy: float = 20e-6,
) -> matplotlib.axes.Axes:
    """
    Plot beam parameters indicated on an area representing the screen. Can be used as an
    overlay.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    sigma_x = max(sigma_x, measurement_accuracy)
    sigma_y = max(sigma_y, measurement_accuracy)

    ax.axvline(mu_x * 1e3, color="lightgrey", ls="--")
    ax.axhline(mu_y * 1e3, color="lightgrey", ls="--")

    sigma_ellipse = Ellipse(
        xy=(mu_x * 1e3, mu_y * 1e3),
        width=6 * sigma_x * 1e3 / 2,
        height=6 * sigma_y * 1e3 / 2,
        facecolor="none",
        edgecolor="lightgrey",
        lw=1,
        ls="--",
    )
    ax.add_patch(sigma_ellipse)

    if resolution is not None and pixel_size is not None:
        (left, right, bottom, top) = screen_extent(resolution, pixel_size * 1e3)
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

    ax.set_title(title)

    if xlabel:
        ax.set_xlabel("(mm)")
    if ylabel:
        ax.set_ylabel("(mm)")

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        fig.savefig(save_path)

    return ax
