from __future__ import annotations

import pickle
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Literal, Optional, Union

import cheetah
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import torch
from scipy.ndimage import uniform_filter

from ..environments.cheetah_env import CheetahEnv

from ..utils import deep_equal
from .utils import (
    parse_problem_index,
    plot_beam_parameters_on_screen,
    plot_screen_image,
)

plt.style.use(["science", "nature", "no-latex"])


class Episode:
    """An episode of an ARES EA optimisation."""

    def __init__(
        self,
        observations: list[Union[dict, np.ndarray]],
        rewards: list[float],
        infos: list[dict],
        actions: list[np.ndarray],
        t_start: Optional[datetime] = None,
        t_end: Optional[datetime] = None,
        steps_taken: Optional[int] = None,
        step_start_times: Optional[list[datetime]] = None,
        step_end_times: Optional[list[datetime]] = None,
        problem_index: Optional[int] = None,
        terminateds: Optional[list[bool]] = None,
        truncateds: Optional[list[bool]] = None,
    ):
        self.observations = observations
        self.rewards = rewards
        self.infos = infos
        self.actions = actions
        self.t_start = t_start
        self.t_end = t_end
        self.steps_taken = steps_taken
        self.step_start_times = step_start_times
        self.step_end_times = step_end_times
        self.problem_index = problem_index
        self.terminateds = terminateds
        self.truncateds = truncateds

        # Fix for old `ares-ea-rl` recordings
        self._add_ea_magnet_names()
        self._fix_backend_info()

    @classmethod
    def load(
        cls,
        path: Union[Path, str],
        use_problem_index: bool = False,
        drop_screen_images: bool = False,
    ) -> Episode:
        """Load the data from one episode recording .pkl file."""
        if isinstance(path, str):
            path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)
        problem_index = parse_problem_index(path) if use_problem_index else None

        loaded = cls(**data, problem_index=problem_index)

        if drop_screen_images:
            loaded.drop_screen_images()

        return loaded

    def save(self, path: Union[Path, str]) -> None:
        """Save the data to a .pkl file."""
        if isinstance(path, str):
            path = Path(path)

        d = {
            "observations": self.observations,
            "rewards": self.rewards,
            "terminateds": self.terminateds,
            "truncateds": self.truncateds,
            "infos": self.infos,
            "actions": self.actions,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "steps_taken": self.steps_taken,
            "step_start_times": self.step_start_times,
            "step_end_times": self.step_end_times,
        }

        with open(path, "wb") as f:
            pickle.dump(d, f)

    def __len__(self) -> int:
        return len(
            self.observations
        )  # Number of steps this episode ran for (including reset)

    def __eq__(self, other: Episode) -> bool:
        attribute_names = [
            "observations",
            "rewards",
            "infos",
            "actions",
            "t_start",
            "t_end",
            "steps_taken",
            "step_start_times",
            "step_end_times",
            "problem_index",
            "terminateds",
            "truncateds",
        ]
        return all(
            deep_equal(getattr(self, name), getattr(other, name))
            for name in attribute_names
        )

    def head(self, n: int) -> Episode:
        """Return an episode with only the first `n` steps of this one."""
        return self.__class__(
            observations=self.observations[:n],
            rewards=self.rewards[:n],
            infos=self.infos[:n],
            actions=self.actions[:n],
            t_start=self.t_start,
            t_end=self.t_end,
            steps_taken=self.steps_taken - 1,
            step_start_times=(
                self.step_start_times[:n] if self.step_start_times else None
            ),
            step_end_times=self.step_end_times[:n] if self.step_end_times else None,
            problem_index=self.problem_index,
            terminateds=self.terminateds[:n] if self.terminateds else None,
            truncateds=self.truncateds[:n] if self.truncateds else None,
        )

    def tail(self, n: int) -> Episode:
        """Return an episode with the last `n` steps of this one."""
        return self.__class__(
            observations=self.observations[-n:],
            rewards=self.rewards[-n:],
            infos=self.infos[-n:],
            actions=self.actions[-n:],
            t_start=self.t_start,
            t_end=self.t_end,
            steps_taken=self.steps_taken - 1,
            step_start_times=(
                self.step_start_times[-n:] if self.step_start_times else None
            ),
            step_end_times=self.step_end_times[-n:] if self.step_end_times else None,
            problem_index=self.problem_index,
            terminateds=self.terminateds[-n:] if self.terminateds else None,
            truncateds=self.truncateds[-n:] if self.truncateds else None,
        )

    def cut_off_at_threshold(self, threshold: float) -> Episode:
        """
        Return an episode with only the steps up to the first step where the MAE is
        below `threshold`.
        """
        n = self.steps_to_threshold(threshold)
        return self.head(n)

    def abs_delta_beam_parameters(self) -> np.ndarray:
        """Get the sequence of mu_x over the episdoe."""
        beams = [obs["beam"] for obs in self.observations]
        target = self.observations[0]["target"]
        abs_deltas = np.abs(np.array(beams) - np.array(target))
        return abs_deltas

    def beam_parameter_history(self) -> np.ndarray:
        """
        Returns the history of the beam parameters. Includes the beam parameters from
        before the reset if there are any.
        """
        beams = np.array([obs["beam"] for obs in self.observations])

        # Check if there are beam parameters from before the reset
        if "beam_before_reset" in self.infos[0]["backend_info"]:
            beam_before_reset = self.infos[0]["backend_info"]["beam_before_reset"]
            beams = np.vstack([beam_before_reset, beams])

        return beams

    def beam_parameters_after(self) -> np.ndarray:
        """
        Get beam parameters from the end of the episode.
        """
        return self.observations[-1]["beam"]

    def beam_parameters_before(self) -> np.ndarray:
        """
        Get diagnostics beam parameters from the start of the episode. If the backend of
        the environment saved beam parameters from before the reset (e.g. the DOOCS
        backend when a magnet initialisation is configured), then those beam parameters
        are returned.
        """
        if "beam_before_reset" in self.infos[0]["backend_info"]:
            return self.infos[0]["backend_info"]["beam_before_reset"]
        else:
            return self.observations[0]["beam"]

    def best_mae(self) -> float:
        """Best MAE observed over the entire duration of the episode."""
        return min(self.maes())

    def drop_screen_images(self) -> None:
        """
        Drop all screen images from this loaded copy of the episode. This can help to
        save RAM while working with the data, when the images are not needed.
        """
        for info in self.infos:
            info.pop("beam_image", None)
            info.pop("screen_before_reset", None)
            info.pop("screen_after_reset", None)

    def final_mae(self) -> float:
        """MAE at the end of the episode."""
        return self.maes()[-1]

    def maes(self, include_before_reset: bool = False) -> np.ndarray:
        """Get the sequence of MAEs over the episdoe."""
        beams = [obs["beam"] for obs in self.observations]
        if include_before_reset:
            beams = [self.infos[0]["backend_info"]["beam_before_reset"]] + beams
        target = self.observations[0]["target"]
        maes = np.mean(np.abs(np.array(beams) - np.array(target)), axis=1)
        return maes

    def magnet_history(self) -> np.ndarray:
        """
        Returns the history of the magnets. Includes the magnet settings from before
        the reset if there are any.
        """
        magnets = np.array([obs["magnets"] for obs in self.observations])

        # Check if there are magnet settings from before the reset
        if "magnets_before_reset" in self.infos[0]["backend_info"]:
            magnets_before_reset = self.infos[0]["backend_info"]["magnets_before_reset"]
            magnets = np.vstack([magnets_before_reset, magnets])

        return magnets

    def min_maes(self, include_before_reset: bool = False) -> np.ndarray:
        """
        Compute the sequences of smallest MAE seen until any given step in the episode.
        """
        maes = self.maes(include_before_reset)
        min_maes = [min(maes[: i + 1]) for i in range(len(maes))]
        return np.array(min_maes)

    def plot_beam_parameters(
        self,
        show_target: bool = True,
        show_target_threshold: Optional[float] = None,
        vertical_marker: Union[float, tuple[float, str]] = None,
        title: Optional[str] = None,
        xlabel: bool = True,
        ylabel: bool = True,
        legend: bool = True,
        mode: str = "all",
        limit_to_screen: bool = True,
        figsize: tuple[float, float] = (6, 3),
        ax: Optional[matplotlib.axes.Axes] = None,
        save_path: Optional[str] = None,
    ) -> matplotlib.axes.Axes:
        """
        Plot beam parameters over the episode and optionally add the target beam
        parameters if `show_target` is `True`. A vertical line to mark a point in time
        may be added via `vertical_marker` either by just its position as a float or by
        a tuple of its position and the string label that should be shown in the legend.
        """
        beams = self.beam_parameter_history()
        targets = self.target_history()

        # palette_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        palette_colors = ["#369F2D", "#EE4431", "#1663A9", "#8E549E"]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if isinstance(vertical_marker, (int, float)):
            ax.axvline(vertical_marker, ls=":", color="grey")
        elif isinstance(vertical_marker, tuple):
            marker_position, marker_label = vertical_marker
            ax.axvline(marker_position, label=marker_label, ls=":", color="grey")
        if mode == "all" or mode == "mu":
            ax.plot(
                np.array(beams)[:, 0] * 1e3,
                label=r"$\mu_x$",
                c=palette_colors[0],
                ls="-",
            )
            ax.plot(
                np.array(beams)[:, 2] * 1e3,
                label=r"$\mu_y$",
                c=palette_colors[2],
                ls="-",
            )
        if mode == "all" or mode == "sigma":
            ax.plot(
                np.array(beams)[:, 1] * 1e3,
                label=r"$\sigma_x$",
                c=palette_colors[1],
                ls="-",
            )
            ax.plot(
                np.array(beams)[:, 3] * 1e3,
                label=r"$\sigma_y$",
                c=palette_colors[3],
                ls="-",
            )

        if show_target:
            if mode == "all" or mode == "mu":
                ax.plot(np.array(targets)[:, 0] * 1e3, c=palette_colors[0], ls="--")
                ax.plot(np.array(targets)[:, 2] * 1e3, c=palette_colors[2], ls="--")
            if mode == "all" or mode == "sigma":
                ax.plot(np.array(targets)[:, 1] * 1e3, c=palette_colors[1], ls="--")
                ax.plot(np.array(targets)[:, 3] * 1e3, c=palette_colors[3], ls="--")

            if show_target_threshold is not None:
                if mode == "all" or mode == "mu":
                    ax.fill_between(
                        np.arange(len(targets)),
                        np.array(targets)[:, 0] * 1e3 - show_target_threshold,
                        np.array(targets)[:, 0] * 1e3 + show_target_threshold,
                        color=palette_colors[0],
                        alpha=0.2,
                    )
                    ax.fill_between(
                        np.arange(len(targets)),
                        np.array(targets)[:, 2] * 1e3 - show_target_threshold,
                        np.array(targets)[:, 2] * 1e3 + show_target_threshold,
                        color=palette_colors[2],
                        alpha=0.2,
                    )
                if mode == "all" or mode == "sigma":
                    ax.fill_between(
                        np.arange(len(targets)),
                        np.array(targets)[:, 1] * 1e3 - show_target_threshold,
                        np.array(targets)[:, 1] * 1e3 + show_target_threshold,
                        color=palette_colors[1],
                        alpha=0.2,
                    )
                    ax.fill_between(
                        np.arange(len(targets)),
                        np.array(targets)[:, 3] * 1e3 - show_target_threshold,
                        np.array(targets)[:, 3] * 1e3 + show_target_threshold,
                        color=palette_colors[3],
                        alpha=0.2,
                    )

        ax.set_title(title)
        if legend:
            legend_cols = 4 if mode == "all" else 2
            ax.legend(loc="upper right", ncol=legend_cols)

        ax.set_xlim(0, None)

        if limit_to_screen:
            # Limit y-axis to screen width (the larger screen side)
            screen_resolution = self.infos[0]["screen_resolution"]
            pixel_size = self.infos[0]["pixel_size"] * 1e3  # Convert to mm
            screen_width = screen_resolution[0] * pixel_size[0]

            if mode == "sigma":
                ax.set_ylim(0 - screen_width / 2 * 0.1, screen_width / 2 * 1.1)
            else:
                ax.set_ylim(-screen_width / 2 * 1.1, screen_width / 2 * 1.1)

        if xlabel:
            ax.set_xlabel("Step")
        if ylabel:
            if mode == "all":
                ax.set_ylabel("Beam parameter (mm)")
            elif mode == "mu":
                ax.set_ylabel("Beam position (mm)")
            elif mode == "sigma":
                ax.set_ylabel("Beam size (mm)")

        if save_path is not None:
            assert fig is not None, "Cannot save figure when axes was passed."
            fig.savefig(save_path)

        return ax

    def plot_best_return_deviation_example(self) -> None:
        """
        Plot an example of MAE over time with markings of the location and value of the
        best setting, to help understand deviations when returning to that setting.
        """
        maes = self.maes()
        first = np.argmin(maes)

        plt.figure(figsize=(5, 3))
        plt.plot(maes)
        plt.axvline(first, c="red")
        plt.axhline(maes[first], c="green")
        plt.show()

    def plot_maes(self, show_best_mae: bool = True) -> None:
        """
        Plot MAE over time. If `show_best_mae` is `True`, add best MAE seen up to a
        certain point to the plot.
        """
        plt.figure(figsize=(6, 3))
        plt.plot(self.maes(), label="MAE")
        if show_best_mae:
            plt.plot(self.min_maes(), label="Best MAE")
            plt.legend()
        plt.ylabel("MAE (m)")
        plt.xlabel("Step")

        plt.show()

    def plot_magnets(
        self,
        vertical_marker: Union[float, tuple[float, str]] = None,
        normalize: bool = False,
        xlabel: bool = True,
        ylabel_left: bool = True,
        ylabel_right: bool = True,
        title: Optional[str] = None,
        legend: bool = True,
        figsize: tuple[float, float] = (6, 3),
        ax: Optional[matplotlib.axes.Axes] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot magnet values over episdoe."""
        magnets = self.magnet_history()

        palette_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if isinstance(vertical_marker, (int, float)):
            ax.axvline(vertical_marker, ls=":", color=palette_colors[5])
        elif isinstance(vertical_marker, tuple):
            marker_position, marker_label = vertical_marker
            ax.axvline(
                marker_position, label=marker_label, ls=":", color=palette_colors[5]
            )

        if normalize:
            ax.plot(magnets[:, 0] / 72.0, c=palette_colors[4], label="Q1")
            ax.plot(magnets[:, 1] / 72.0, c=palette_colors[5], label="Q2")
            ax.plot(magnets[:, 2] / 6.1782e-3, c=palette_colors[6], label="CV")
            ax.plot(magnets[:, 3] / 72.0, c="#ee3377", label="Q3")
            ax.plot(magnets[:, 4] / 6.1782e-3, c="#33bbee", label="CH")
            ax.set_xlim(0, None)
            ax.set_ylim(-1.1, 1.1)
            if legend:
                ax.legend()

            if ylabel_left:
                ax.set_ylabel("Normalised actuator setting")

        else:
            ax.plot(magnets[:, 0], c=palette_colors[4], label="Q1")
            ax.plot(magnets[:, 1], c=palette_colors[5], label="Q2")
            ax.plot([], c=palette_colors[6], label="CV")  # Dummy for legend
            ax.plot(magnets[:, 3], c="#ee3377", label="Q3")
            ax.plot([], c="#33bbee", label="CH")  # Dummy for legend
            ax.set_xlim(0, None)
            ax.set_ylim(-72 * 1.1, 72 * 1.1)
            if legend:
                ax.legend()

            ax_twinx = ax.twinx()

            ax_twinx.plot(magnets[:, 2] * 1e3, c=palette_colors[6], label="CV")
            ax_twinx.plot(magnets[:, 4] * 1e3, c="#33bbee", label="CH")
            ax_twinx.set_ylabel("Steering Angle (mrad)")
            ax_twinx.set_ylim(-6.1782 * 1.1, 6.1782 * 1.1)

            if ylabel_left:
                ax.set_ylabel(r"Quadrupole strength ($m^{-2}$)")
            if ylabel_right:
                ax_twinx.set_ylabel("Steering angle (mrad)")

        ax.set_title(title)

        if xlabel:
            ax.set_xlabel("Step")

        if save_path is not None:
            assert fig is not None, "Cannot save figure when axes was passed."
            fig.savefig(save_path)

        return ax

    def plot_misalignments(
        self,
        vertical_marker: Union[float, tuple[float, str]] = None,
        add_incoming_position: bool = False,
        title: Optional[str] = None,
        xlabel: bool = True,
        ylabel: bool = True,
        legend: bool = True,
        figsize: tuple[float, float] = (6, 3),
        ax: Optional[matplotlib.axes.Axes] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot the misalignments of the quadrupoles and the screen over the episode."""
        misalignments = np.array(
            [info["backend_info"]["misalignments"] for info in self.infos]
        )

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if isinstance(vertical_marker, (int, float)):
            ax.axvline(vertical_marker, ls=":", color="grey")
        elif isinstance(vertical_marker, tuple):
            marker_position, marker_label = vertical_marker
            ax.axvline(marker_position, label=marker_label, ls=":", color="grey")

        quadrupole_names = [
            name for name in self.infos[0]["magnet_names"] if name[5] == "Q"
        ]
        screen_name = self.infos[0]["screen_name"]

        for i, name in enumerate(quadrupole_names):
            ax.plot(misalignments[:, i * 2], label=f"{name} x")
            ax.plot(misalignments[:, i * 2 + 1], label=f"{name} y")
        ax.plot(misalignments[:, -2], label=f"{screen_name} x")
        ax.plot(misalignments[:, -1], label=f"{screen_name} y")

        if add_incoming_position:
            incoming_beams = np.array(
                [info["backend_info"]["incoming_beam"] for info in self.infos]
            )
            ax.plot(incoming_beams[:, 1], label="Incoming x", ls="--")
            ax.plot(incoming_beams[:, 3], label="Incoming y", ls="--")

        ax.set_ylim(5e-4 * -1.1, 5e-4 * 1.1)

        ax.set_title(title)
        if legend:
            ax.legend(loc="upper right", ncol=2)
        if xlabel:
            ax.set_xlabel("Step")
        if ylabel:
            ax.set_ylabel("Misalignment and incoming position (mm)")

        if save_path is not None:
            assert fig is not None, "Cannot save figure when axes was passed."
            fig.savefig(save_path)

        return ax

    def plot_quadrupoles(
        self,
        vertical_marker: Union[float, tuple[float, str]] = None,
        title: Optional[str] = None,
        normalize: bool = False,
        xlabel: bool = True,
        ylabel: bool = True,
        legend: bool = True,
        figsize: tuple[float, float] = (6, 3),
        ax: Optional[matplotlib.axes.Axes] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot quadrupole strengths over episode."""
        all_histories = self.quadrupole_history()

        palette_colors = ["#EC748B", "#C4A751", "#36ACA2"]
        assert all_histories.shape[1] <= len(
            palette_colors
        ), "Not enough colours for all quadrupoles."

        names = [
            name
            for name in self.infos[0]["magnet_names"]
            if name[0] == "Q" or (len(name) > 5 and name[5] == "Q")
        ]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if isinstance(vertical_marker, (int, float)):
            ax.axvline(vertical_marker, ls=":", color="grey")
        elif isinstance(vertical_marker, tuple):
            marker_position, marker_label = vertical_marker
            ax.axvline(marker_position, label=marker_label, ls=":", color="grey")

        max_quad_setting = self.infos[0]["max_quad_setting"]

        if normalize:
            for name, history, color in zip(names, all_histories.T, palette_colors):
                ax.plot(history / max_quad_setting, c=color, label=name, ls="-")
            ax.set_xlim(0, None)
            ax.set_ylim(-1.1, 1.1)
            if legend:
                ax.legend(loc="upper right", ncol=len(names))

            if ylabel:
                ax.set_ylabel("Normalised actuator setting")

        else:
            for name, history, color in zip(names, all_histories.T, palette_colors):
                ax.plot(history, c=color, label=name, ls="-")
            ax.set_xlim(0, None)
            ax.set_ylim(
                -max_quad_setting * 1.1, max_quad_setting * 1.1
            )  # Limits + 10% margin
            if legend:
                ax.legend(loc="upper right", ncol=len(names))

            if ylabel:
                ax.set_ylabel(r"Quadrupole Strength ($m^{-2}$)")

        ax.set_title(title)

        if xlabel:
            ax.set_xlabel("Step")
        if save_path is not None:
            assert fig is not None, "Cannot save figure when axes was passed."
            fig.savefig(save_path)

        return ax

    def plot_rewards(
        self,
        vertical_marker: Union[float, tuple[float, str]] = None,
        title: Optional[str] = None,
        xlabel: bool = True,
        ylabel: bool = True,
        legend: bool = True,
        figsize: tuple[float, float] = (6, 3),
        ax: Optional[matplotlib.axes.Axes] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot the rewards and its components over the episode."""
        rewards = np.array(self.rewards)
        beam_rewards = np.array([info["beam_reward"] for info in self.infos])
        on_screen_rewards = np.array([info["on_screen_reward"] for info in self.infos])
        magnet_change_rewards = np.array(
            [info["magnet_change_reward"] for info in self.infos]
        )

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if isinstance(vertical_marker, (int, float)):
            ax.axvline(vertical_marker, ls=":", color="grey")
        elif isinstance(vertical_marker, tuple):
            marker_position, marker_label = vertical_marker
            ax.axvline(marker_position, label=marker_label, ls=":", color="grey")

        ax.plot(rewards, label="Total")
        ax.plot(beam_rewards, label="Beam")
        ax.plot(on_screen_rewards, label="On screen")
        ax.plot(magnet_change_rewards, label="Magnet changes")

        ax.set_ylim(0, 1.1)

        if legend:
            ax.legend()

        ax.set_title(title)

        if xlabel:
            ax.set_xlabel("Step")
        if ylabel:
            ax.set_ylabel("Reward")

        if save_path is not None:
            assert fig is not None, "Cannot save figure when axes was passed."
            fig.savefig(save_path)

        return ax

    def plot_steerers(
        self,
        vertical_marker: Union[float, tuple[float, str]] = None,
        title: Optional[str] = None,
        normalize: bool = False,
        xlabel: bool = True,
        ylabel: bool = True,
        legend: bool = True,
        figsize: tuple[float, float] = (6, 3),
        ax: Optional[matplotlib.axes.Axes] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot steerings angles over episdoe."""
        all_histories = self.steerer_history()

        palette_colors = ["#9e9e9e", "#33bbee"]
        assert all_histories.shape[1] <= len(
            palette_colors
        ), "Not enough colours for all steerers."

        names = [
            name
            for name in self.infos[0]["magnet_names"]
            if name[0] == "C" or (len(name) > 5 and name[5] == "C")
        ]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if isinstance(vertical_marker, (int, float)):
            ax.axvline(vertical_marker, ls=":", color="grey")
        elif isinstance(vertical_marker, tuple):
            marker_position, marker_label = vertical_marker
            ax.axvline(marker_position, label=marker_label, ls=":", color="grey")

        if normalize:
            for name, history, color in zip(names, all_histories.T, palette_colors):
                ax.plot(history / 6.1782e-3, c=color, label=name, ls="-")
            ax.set_xlim(0, None)
            ax.set_ylim(-1.1, 1.1)
            if legend:
                ax.legend(loc="upper right", ncol=len(names))

            if ylabel:
                ax.set_ylabel("Normalised actuator setting")

        else:
            for name, history, color in zip(names, all_histories.T, palette_colors):
                ax.plot(history * 1e3, c=color, label=name, ls="-")
            ax.set_xlim(0, None)
            ax.set_ylim(-6.1782, 6.1782)  # Limits + 10% margin
            if legend:
                ax.legend(loc="upper right", ncol=len(names))

            if ylabel:
                ax.set_ylabel("Steering Angle (mrad)")

        ax.set_title(title)

        if xlabel:
            ax.set_xlabel("Step")
        if save_path is not None:
            assert fig is not None, "Cannot save figure when axes was passed."
            fig.savefig(save_path)

        return ax

    def plot_summary(
        self,
        title: Optional[str] = None,
        figsize: tuple[float, float] = (510 / 72.72 * 1.9, 510 / 72.72 * 1.205 * 0.45),
        show_domain_randomisation_and_rewards: bool = False,
        fake_screen_images: bool = True,
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """Summary plot of important data about this episode."""

        if show_domain_randomisation_and_rewards:
            figsize = (figsize[0], figsize[1] * 1.5)

        # Plot Layout
        fig = plt.figure(figsize=figsize)
        grid_heigth = 3 if show_domain_randomisation_and_rewards else 2
        gs = fig.add_gridspec(grid_heigth, 3, width_ratios=[1, 1, 0.67], hspace=0.1)

        ax_steerer = fig.add_subplot(gs[0, 0])
        ax_quadrupole = fig.add_subplot(gs[1, 0], sharex=ax_steerer)
        ax_mu = fig.add_subplot(gs[0, 1], sharex=ax_steerer)
        ax_sigma = fig.add_subplot(gs[1, 1], sharex=ax_steerer)

        ax_before = fig.add_subplot(gs[0, 2])
        ax_after = fig.add_subplot(gs[1, 2], sharex=ax_before, sharey=ax_before)

        for ax in [ax_steerer, ax_mu, ax_before]:
            ax.xaxis.set_tick_params(labelbottom=False)

        #############
        # Plotting
        self.plot_quadrupoles(
            ax=ax_quadrupole,
            xlabel=not show_domain_randomisation_and_rewards,
            ylabel=True,
            legend=True,
            normalize=False,
        )

        self.plot_steerers(
            ax=ax_steerer,
            xlabel=False,
            ylabel=True,
            legend=True,
            normalize=False,
            title="Magnet settings",
        )

        self.plot_beam_parameters(
            ax=ax_mu,
            xlabel=False,
            ylabel=True,
            mode="mu",
            legend=True,
            title="Beam parameters",
            limit_to_screen=False,
        )

        self.plot_beam_parameters(
            ax=ax_sigma,
            xlabel=not show_domain_randomisation_and_rewards,
            ylabel=True,
            mode="sigma",
            legend=True,
            limit_to_screen=False,
        )

        #############
        # Screen Images
        if "screen_image" in self.infos[0]["backend_info"]:
            plot_screen_image(
                image=self.screen_image_before(),
                resolution=self.infos[0]["screen_resolution"],
                pixel_size=self.infos[0]["pixel_size"],
                ax=ax_before,
                vmax=self.screen_image_shared_vmax(),
                xlabel=False,
                ylabel=True,
                title="Screen images",
            )
            plot_screen_image(
                image=self.screen_image_after(),
                resolution=self.infos[-1]["screen_resolution"],
                pixel_size=self.infos[-1]["pixel_size"],
                ax=ax_after,
                vmax=self.screen_image_shared_vmax(),
                xlabel=True,
                ylabel=True,
            )
        elif fake_screen_images:
            fake_screen = cheetah.Screen(
                resolution=(
                    int(self.infos[0]["screen_resolution"][0]),
                    int(self.infos[0]["screen_resolution"][1]),
                ),
                pixel_size=torch.tensor(self.infos[0]["pixel_size"]),
                is_active=True,
            )

            fake_beam_before = cheetah.ParameterBeam.from_parameters(
                mu_x=torch.tensor(self.beam_parameters_before()[0]),
                sigma_x=torch.tensor(self.beam_parameters_before()[1]),
                mu_y=torch.tensor(self.beam_parameters_before()[2]),
                sigma_y=torch.tensor(self.beam_parameters_before()[3]),
            )
            _ = fake_screen.track(fake_beam_before)
            fake_screen_image_before = fake_screen.reading.numpy()

            fake_beam_after = cheetah.ParameterBeam.from_parameters(
                mu_x=torch.tensor(self.beam_parameters_after()[0]),
                sigma_x=torch.tensor(self.beam_parameters_after()[1]),
                mu_y=torch.tensor(self.beam_parameters_after()[2]),
                sigma_y=torch.tensor(self.beam_parameters_after()[3]),
            )
            _ = fake_screen.track(fake_beam_after)
            fake_screen_image_after = fake_screen.reading.numpy()

            fake_screen_image_shared_vmax = max(
                fake_screen_image_before.max(), fake_screen_image_after.max()
            )

            plot_screen_image(
                image=fake_screen_image_before,
                resolution=self.infos[0]["screen_resolution"],
                pixel_size=self.infos[0]["pixel_size"],
                ax=ax_before,
                vmax=fake_screen_image_shared_vmax,
                xlabel=False,
                ylabel=True,
                title="Screen images",
            )
            plot_screen_image(
                image=fake_screen_image_after,
                resolution=self.infos[-1]["screen_resolution"],
                pixel_size=self.infos[-1]["pixel_size"],
                ax=ax_after,
                vmax=fake_screen_image_shared_vmax,
                xlabel=True,
                ylabel=True,
            )

        plot_beam_parameters_on_screen(
            mu_x=self.target_beam_parameters()[0],
            sigma_x=self.target_beam_parameters()[1],
            mu_y=self.target_beam_parameters()[2],
            sigma_y=self.target_beam_parameters()[3],
            resolution=self.infos[0]["screen_resolution"],
            pixel_size=self.infos[0]["pixel_size"],
            ax=ax_after,
            xlabel=False,
            ylabel=False,
        )

        ax_before.text(0.05, 0.85, "Before", transform=ax_before.transAxes)
        ax_after.text(0.05, 0.85, "After", transform=ax_after.transAxes)

        # Fine tuning
        fig.align_ylabels([ax_steerer, ax_quadrupole])
        fig.align_ylabels([ax_mu, ax_sigma])

        # Optional domain randomisation and rewards
        if show_domain_randomisation_and_rewards:
            ax_domain_randomisation = fig.add_subplot(gs[2, 0], sharex=ax_steerer)
            ax_rewards = fig.add_subplot(gs[2, 1], sharex=ax_steerer)

            self.plot_misalignments(
                add_incoming_position=True,
                ax=ax_domain_randomisation,
                xlabel=True,
                ylabel=True,
                legend=True,
                title="Domain randomisation",
            )
            self.plot_rewards(
                ax=ax_rewards,
                xlabel=True,
                ylabel=True,
                legend=True,
                title="Rewards",
            )

        fig.suptitle(title)

        if save_path is not None:
            fig.savefig(save_path)

        return fig

    def quadrupole_history(self) -> np.ndarray:
        """
        Returns only the columns from the `magnet_history` that correspond to
        quadrupoles.
        """
        magnets = self.magnet_history()
        magnet_names = self.infos[0]["magnet_names"]
        quadrupole_indices = [
            i
            for i, name in enumerate(magnet_names)
            if name[0] == "Q" or (len(name) > 5 and name[5] == "Q")
        ]
        return magnets[:, quadrupole_indices]

    def rmse(self, use_best_beam: bool = False) -> float:
        """
        RMSE over all samples in episode and over all beam parameters as used in
        https://www.nature.com/articles/s41586-021-04301-9.
        """
        beams = (
            self.best_beam_history()
            if use_best_beam
            else np.stack([obs["beam"] for obs in self.observations])
        )
        targets = np.stack([obs["target"] for obs in self.observations])
        rmse = np.sqrt(np.mean(np.square(targets - beams)))
        return rmse

    def accumulated_mae(self, use_min_mae: bool = False) -> float:
        """
        Takes the mean of all MAEs over the episode.
        """
        return self.maes().mean() if not use_min_mae else self.min_maes().mean()

    def normalized_accumulated_mae(self, use_min_mae: bool = False) -> float:
        """
        Takes the mean of all MAEs over the episode and divides it by initial target
        MAE.
        """
        return self.accumulated_mae(use_min_mae) / self.maes()[0]

    def mae_improvement(
        self, use_min_mae: bool = False, include_before_reset: bool = False
    ) -> float:
        """
        Compute the improvement in MAE over the episode. Positive values indicate
        improvement, negative values indicate worsening.
        """
        maes = (
            self.min_maes(include_before_reset)
            if use_min_mae
            else self.maes(include_before_reset)
        )
        return maes[0] - maes[-1]

    def normalized_mae_improvement(
        self, use_min_mae: bool = False, include_before_reset: bool = False
    ) -> float:
        """
        Compute the improvement in MAE over the episode normalised to the initial MAE.
        Positive values indicate improvement, negative values indicate worsening.
        """
        improvement = self.mae_improvement(
            use_min_mae=use_min_mae, include_before_reset=include_before_reset
        )
        return improvement / self.maes(include_before_reset)[0]

    def screen_image_after(self):
        """
        Get diagnostics screen image from the end of the episode.
        """
        return self.infos[-1]["backend_info"]["screen_image"]

    def screen_image_before(self) -> np.ndarray:
        """
        Get diagnostics screen image from the start of the episode. If the backend of
        the environment saved a screen image from before the reset (e.g. the DOOCS
        backend when a magnet initialisation is configured), then this image is
        returned.
        """
        if "screen_before_reset" in self.infos[0]:
            return self.infos[0]["backend_info"]["screen_before_reset"]
        else:
            return self.infos[0]["backend_info"]["screen_image"]

    def screen_image_shared_vmax(self) -> float:
        """
        Compute the highest value accross all (processed) screen images recorded over
        the episdoe.
        """
        if "screen_before_reset" in self.infos[0]["backend_info"]:
            screen_images = [self.infos[0]["backend_info"]["screen_before_reset"]]
        else:
            screen_images = []
        screen_images += [info["backend_info"]["screen_image"] for info in self.infos]

        # Process images
        processed_images = [uniform_filter(image, size=10) for image in screen_images]

        return np.max(np.stack(processed_images))

    def steerer_history(self) -> np.ndarray:
        """
        Returns only the columns from the `magnet_history` that correspond to steerers.
        """
        magnets = self.magnet_history()
        magnet_names = self.infos[0]["magnet_names"]
        steerer_indices = [
            i
            for i, name in enumerate(magnet_names)
            if name[0] == "C" or (len(name) > 5 and name[5] == "C")
        ]
        return magnets[:, steerer_indices]

    def steps_to_convergence(
        self, threshold: float = 20e-6, use_min_mae: bool = False
    ) -> int:
        """
        Find the number of steps until the MAEs converge towards some value, i.e. change
        no more than threshold in the future. If no convergence is found before the last
        step, the total number of steps in the episode is returned.
        """
        maes = self.min_maes() if use_min_mae else self.maes()

        # Start at 1 because 0s can cause KDE on the results to fail
        convergence_step = 1
        for i in range(1, len(maes)):
            convergence_step = i
            maes_from_now_on = maes[i:]
            if max(maes_from_now_on) - min(maes_from_now_on) < threshold:
                break

        return convergence_step

    def steps_to_threshold(
        self,
        threshold: float = 20e-6,
        use_min_mae: bool = False,
        allow_lowest_as_target: bool = False,
    ) -> int:
        """
        Find the number of steps until the maes in `episdoe` drop below `threshold`. If
        `allow_lowest_as_target` is `True` and the threshold was never reached, the
        number of steps until the lowest MAE is returned, otherwise return the number of
        total steps in the episode.
        """
        maes = np.array(self.min_maes() if use_min_mae else self.maes())
        arg_lower = np.argwhere(maes < threshold).squeeze()
        if len(arg_lower.shape) == 0:  # 0-dimensional one number
            return int(arg_lower)
        elif len(arg_lower) == 0:  # 0 elements in 1-dimensional array (no result)
            return np.argmin(maes) if allow_lowest_as_target else len(maes)
        else:
            return arg_lower[0]

    @property
    def target(self) -> np.ndarray:
        return self.observations[-1]["target"]

    def target_size(self) -> float:
        """Compute a measure of size for the episode's target."""
        return np.mean([self.target[1], self.target[3]])

    def target_beam_parameters(self) -> np.ndarray:
        """
        Get the target beam parameters for this episode.
        """
        return self.observations[0]["target"]

    def target_history(self) -> np.ndarray:
        """
        Returns the history of the target beam parameters. Padds the start with NaNs if
        there are beam parameters from before the reset.
        """
        targets = np.array([obs["target"] for obs in self.observations])

        # Check if there are beam parameters from before the reset
        if "beam_before_reset" in self.infos[0]["backend_info"]:
            targets = np.vstack([np.full(4, np.nan), targets])

        return targets

    def error_history(self) -> np.ndarray:
        """
        Returns the history of the error between the target and the beam parameters. If
        there are beam parameters from before the reset, the first error computed to the
        target at reset.
        """
        beams = self.beam_parameter_history()
        targets = self.target_history()

        if len(beams) > len(targets):
            targets = np.vstack([targets[0], targets])

        errors = np.abs(beams - targets)

        return errors

    def best_beam_parameter_error(
        self, name: Literal["mu_x", "sigma_x", "mu_y", "sigma_y"]
    ) -> float:
        """
        Return the best value of a beam parameter compared to the target over the
        episode.
        """
        parameter_index = {"mu_x": 0, "sigma_x": 1, "mu_y": 2, "sigma_y": 3}[name]

        errors = self.error_history()
        parameter_errors = errors[:, parameter_index]

        return np.nanmin(parameter_errors)

    def best_magnet_settings(self) -> np.ndarray:
        """
        Return the best magnet settings resulting in the best MAE from this episode.
        """
        maes = self.maes()
        best_index = np.argmin(maes)
        return self.observations[best_index]["magnets"]

    def best_beam_history(self) -> np.ndarray:
        """
        Return the beam parameters of the best found solution at each step.
        """
        beams = np.array([obs["beam"] for obs in self.observations])
        maes = self.maes()

        best_beams = np.full_like(beams, np.nan)
        for i in range(beams.shape[0]):
            min_mae_index = np.argmin(maes[: i + 1])
            best_beams[i] = beams[min_mae_index]

        return best_beams

    def plot_objective_in_space(
        self,
        fig: Optional[matplotlib.figure.Figure] = None,
        figsize: tuple[float, float] = (8, 8),
        assumed_misalignments: Optional[np.ndarray] = None,
        assumed_incoming_beam: Optional[np.ndarray] = None,
        num_objective_samples: int = 20,
        cmap: str = "viridis",
    ) -> matplotlib.figure.Figure:
        """
        Plot the samples of the episode in the objective space slices of the best found
        solution.
        """
        if fig is None:
            fig, axs = plt.subplots(4, 4, sharex="col", sharey="row", figsize=figsize)
        else:
            axs = fig.subplots(4, 4, sharex="col", sharey="row")

        episode = self.cut_off_at_threshold(threshold=4e-5)
        best_settings = episode.best_magnet_settings()

        assumed_misalignments = (
            self.infos[0]["backend_info"]["misalignments"]
            if assumed_misalignments is None
            else assumed_misalignments
        )
        assumed_incoming_beam = (
            self.infos[0]["backend_info"]["incoming_beam"]
            if assumed_incoming_beam is None
            else assumed_incoming_beam
        )

        # Visualise space
        env = cheetah_env.CheetahEnv(
            action_mode="direct",
            clip_magnets=True,
            backend="cheetah",
            incoming_mode=assumed_incoming_beam,
            misalignment_mode=assumed_misalignments,
        )
        _, _ = env.reset()

        quad_samples = np.linspace(-30, 30, num_objective_samples)
        steerer_samples = np.linspace(-6.1782e-3, 6.1782e-3, num_objective_samples)

        def get_objective_value(magnets: np.ndarray) -> float:
            observation, _, _, _, _ = env.step(magnets)
            objective = np.log(
                np.mean(np.abs(observation["beam"] - self.observations[0]["target"]))
            )
            return objective

        axs_3_0_img_x, axs_3_0_img_y = np.meshgrid(quad_samples, steerer_samples)
        objectives = np.full((num_objective_samples, num_objective_samples), np.nan)
        for i, j in product(range(num_objective_samples), range(num_objective_samples)):
            action = best_settings.copy()
            action[0], action[4] = axs_3_0_img_x[i, j], axs_3_0_img_y[i, j]
            objectives[i, j] = get_objective_value(action)
        axs[3, 0].imshow(
            objectives,
            aspect="auto",
            origin="lower",
            extent=[-30 / 72, 30 / 72, -1, 1],
            cmap=cmap,
        )

        axs_3_1_img_x, axs_3_1_img_y = np.meshgrid(quad_samples, steerer_samples)
        objectives = np.full((num_objective_samples, num_objective_samples), np.nan)
        for i, j in product(range(num_objective_samples), range(num_objective_samples)):
            action = best_settings.copy()
            action[1], action[4] = axs_3_1_img_x[i, j], axs_3_1_img_y[i, j]
            objectives[i, j] = get_objective_value(action)
        axs[3, 1].imshow(
            objectives,
            aspect="auto",
            origin="lower",
            extent=[-30 / 72, 30 / 72, -1, 1],
            cmap=cmap,
        )

        axs_3_2_img_x, axs_3_2_img_y = np.meshgrid(steerer_samples, steerer_samples)
        objectives = np.full((num_objective_samples, num_objective_samples), np.nan)
        for i, j in product(range(num_objective_samples), range(num_objective_samples)):
            action = best_settings.copy()
            action[2], action[4] = axs_3_2_img_x[i, j], axs_3_2_img_y[i, j]
            objectives[i, j] = get_objective_value(action)
        axs[3, 2].imshow(
            objectives, aspect="auto", origin="lower", extent=[-1, 1, -1, 1], cmap=cmap
        )

        axs_3_3_img_x, axs_3_3_img_y = np.meshgrid(quad_samples, steerer_samples)
        objectives = np.full((num_objective_samples, num_objective_samples), np.nan)
        for i, j in product(range(num_objective_samples), range(num_objective_samples)):
            action = best_settings.copy()
            action[3], action[4] = axs_3_3_img_x[i, j], axs_3_3_img_y[i, j]
            objectives[i, j] = get_objective_value(action)
        axs[3, 3].imshow(
            objectives,
            aspect="auto",
            origin="lower",
            extent=[-30 / 72, 30 / 72, -1, 1],
            cmap=cmap,
        )

        axs_2_0_img_x, axs_2_0_img_y = np.meshgrid(quad_samples, quad_samples)
        objectives = np.full((num_objective_samples, num_objective_samples), np.nan)
        for i, j in product(range(num_objective_samples), range(num_objective_samples)):
            action = best_settings.copy()
            action[0], action[3] = axs_2_0_img_x[i, j], axs_2_0_img_y[i, j]
            objectives[i, j] = get_objective_value(action)
        axs[2, 0].imshow(
            objectives,
            aspect="auto",
            origin="lower",
            extent=[-30 / 72, 30 / 72, -30 / 72, 30 / 72],
            cmap=cmap,
        )

        axs_2_1_img_x, axs_2_1_img_y = np.meshgrid(quad_samples, quad_samples)
        objectives = np.full((num_objective_samples, num_objective_samples), np.nan)
        for i, j in product(range(num_objective_samples), range(num_objective_samples)):
            action = best_settings.copy()
            action[1], action[3] = axs_2_1_img_x[i, j], axs_2_1_img_y[i, j]
            objectives[i, j] = get_objective_value(action)
        axs[2, 1].imshow(
            objectives,
            aspect="auto",
            origin="lower",
            extent=[-30 / 72, 30 / 72, -30 / 72, 30 / 72],
            cmap=cmap,
        )

        axs_2_2_img_x, axs_2_2_img_y = np.meshgrid(steerer_samples, quad_samples)
        objectives = np.full((num_objective_samples, num_objective_samples), np.nan)
        for i, j in product(range(num_objective_samples), range(num_objective_samples)):
            action = best_settings.copy()
            action[2], action[3] = axs_2_2_img_x[i, j], axs_2_2_img_y[i, j]
            objectives[i, j] = get_objective_value(action)
        axs[2, 2].imshow(
            objectives,
            aspect="auto",
            origin="lower",
            extent=[-1, 1, -30 / 72, 30 / 72],
            cmap=cmap,
        )

        axs_1_0_img_x, axs_1_0_img_y = np.meshgrid(quad_samples, steerer_samples)
        objectives = np.full((num_objective_samples, num_objective_samples), np.nan)
        for i, j in product(range(num_objective_samples), range(num_objective_samples)):
            action = best_settings.copy()
            action[0], action[2] = axs_1_0_img_x[i, j], axs_1_0_img_y[i, j]
            objectives[i, j] = get_objective_value(action)
        axs[1, 0].imshow(
            objectives,
            aspect="auto",
            origin="lower",
            extent=[-30 / 72, 30 / 72, -1, 1],
            cmap=cmap,
        )

        axs_1_1_img_x, axs_1_1_img_y = np.meshgrid(quad_samples, steerer_samples)
        objectives = np.full((num_objective_samples, num_objective_samples), np.nan)
        for i, j in product(range(num_objective_samples), range(num_objective_samples)):
            action = best_settings.copy()
            action[1], action[2] = axs_1_1_img_x[i, j], axs_1_1_img_y[i, j]
            objectives[i, j] = get_objective_value(action)
        axs[1, 1].imshow(
            objectives,
            aspect="auto",
            origin="lower",
            extent=[-30 / 72, 30 / 72, -1, 1],
            cmap=cmap,
        )

        axs_0_0_img_x, axs_0_0_img_y = np.meshgrid(quad_samples, quad_samples)
        objectives = np.full((num_objective_samples, num_objective_samples), np.nan)
        for i, j in product(range(num_objective_samples), range(num_objective_samples)):
            action = best_settings.copy()
            action[0], action[1] = axs_0_0_img_x[i, j], axs_0_0_img_y[i, j]
            objectives[i, j] = get_objective_value(action)
        axs[0, 0].imshow(
            objectives,
            aspect="auto",
            origin="lower",
            extent=[-30 / 72, 30 / 72, -30 / 72, 30 / 72],
            cmap=cmap,
        )

        # Plot episode samples
        axs[3, 0].scatter(
            [obs["magnets"][0] / 72 for obs in episode.observations],
            [obs["magnets"][4] / 6.1782e-3 for obs in episode.observations],
            color="black",
            s=0.8,
        )
        axs[3, 0].scatter(
            [best_settings[0] / 72],
            [best_settings[4] / 6.1782e-3],
            c="red",
            marker="*",
            s=20,
        )
        axs[3, 0].set_xlabel(r"$Q_1$")
        axs[3, 0].set_ylabel(r"$C_h$")

        axs[3, 1].scatter(
            [obs["magnets"][1] / 72 for obs in episode.observations],
            [obs["magnets"][4] / 6.1782e-3 for obs in episode.observations],
            color="black",
            s=0.8,
        )
        axs[3, 1].scatter(
            [best_settings[1] / 72],
            [best_settings[4] / 6.1782e-3],
            c="red",
            marker="*",
            s=20,
        )
        axs[3, 1].set_xlabel(r"$Q_2$")

        axs[3, 2].scatter(
            [obs["magnets"][2] / 6.1782e-3 for obs in episode.observations],
            [obs["magnets"][4] / 6.1782e-3 for obs in episode.observations],
            color="black",
            s=0.8,
        )
        axs[3, 2].scatter(
            [best_settings[2] / 6.1782e-3],
            [best_settings[4] / 6.1782e-3],
            c="red",
            marker="*",
            s=20,
        )
        axs[3, 2].set_xlabel(r"$C_v$")

        axs[3, 3].scatter(
            [obs["magnets"][3] / 72 for obs in episode.observations],
            [obs["magnets"][4] / 6.1782e-3 for obs in episode.observations],
            color="black",
            s=0.8,
        )
        axs[3, 3].scatter(
            [best_settings[3] / 72],
            [best_settings[4] / 6.1782e-3],
            c="red",
            marker="*",
            s=20,
        )
        axs[3, 3].set_xlabel(r"$Q_3$")

        axs[2, 0].scatter(
            [obs["magnets"][0] / 72 for obs in episode.observations],
            [obs["magnets"][3] / 72 for obs in episode.observations],
            color="black",
            s=0.8,
        )
        axs[2, 0].scatter(
            [best_settings[0] / 72], [best_settings[3] / 72], c="red", marker="*", s=20
        )
        axs[2, 0].set_ylabel(r"$Q_3$")

        axs[2, 1].scatter(
            [obs["magnets"][1] / 72 for obs in episode.observations],
            [obs["magnets"][3] / 72 for obs in episode.observations],
            color="black",
            s=0.8,
        )
        axs[2, 1].scatter(
            [best_settings[1] / 72], [best_settings[3] / 72], c="red", marker="*", s=20
        )

        axs[2, 2].scatter(
            [obs["magnets"][2] / 6.1782e-3 for obs in episode.observations],
            [obs["magnets"][3] / 72 for obs in episode.observations],
            color="black",
            s=0.8,
        )
        axs[2, 2].scatter(
            [best_settings[2] / 6.1782e-3],
            [best_settings[3] / 72],
            c="red",
            marker="*",
            s=20,
        )

        axs[1, 0].scatter(
            [obs["magnets"][0] / 72 for obs in episode.observations],
            [obs["magnets"][2] / 6.1782e-3 for obs in episode.observations],
            color="black",
            s=0.8,
        )
        axs[1, 0].scatter(
            [best_settings[0] / 72],
            [best_settings[2] / 6.1782e-3],
            c="red",
            marker="*",
            s=20,
        )
        axs[1, 0].set_ylabel(r"$C_v$")

        axs[1, 1].scatter(
            [obs["magnets"][1] / 72 for obs in episode.observations],
            [obs["magnets"][2] / 6.1782e-3 for obs in episode.observations],
            color="black",
            s=0.8,
        )
        axs[1, 1].scatter(
            [best_settings[1] / 72],
            [best_settings[2] / 6.1782e-3],
            c="red",
            marker="*",
            s=20,
        )

        axs[0, 0].scatter(
            [obs["magnets"][0] / 72 for obs in episode.observations],
            [obs["magnets"][1] / 72 for obs in episode.observations],
            color="black",
            s=0.8,
        )
        axs[0, 0].scatter(
            [best_settings[0] / 72], [best_settings[1] / 72], c="red", marker="*", s=20
        )
        axs[0, 0].set_ylabel(r"$Q_2$")

        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        axs[3, 0].set_xlim(-30 / 72, 30 / 72)
        axs[3, 1].set_xlim(-30 / 72, 30 / 72)
        axs[3, 2].set_xlim(-1, 1)
        axs[3, 3].set_xlim(-30 / 72, 30 / 72)
        axs[2, 0].set_ylim(-30 / 72, 30 / 72)
        axs[1, 0].set_ylim(-1, 1)
        axs[0, 0].set_ylim(-30 / 72, 30 / 72)

        axs[0, 1].axis("off")
        axs[0, 2].axis("off")
        axs[0, 3].axis("off")
        axs[1, 2].axis("off")
        axs[1, 3].axis("off")
        axs[2, 3].axis("off")

        return fig

    def _fix_backend_info(self):
        """
        Old recordings done with `ares-ea-rl` don't have a separate `backend_info`. This
        method fixes that by moving the fields that belong in `backend_info` from the
        top level of `info` to `backend_info`.
        """
        for info in self.infos:
            if "backend_info" not in info:
                info["backend_info"] = {}
                for key in [
                    "screen_image",
                    "camera_gain",
                    "ARLIMCHM1",
                    "ARLIMCVM1",
                    "ARLIMCHM2",
                    "ARLIMCVM2",
                    "gun_solenoid",
                    "beam_before_reset",
                    "screen_before_reset",
                    "magnets_before_reset",
                    "incoming_beam",
                    "misalignments",
                ]:
                    if key in info:
                        info["backend_info"][key] = info[key]
                        del info[key]

                if "beam_image" in info:
                    info["backend_info"]["screen_image"] = info["beam_image"]
                    del info["beam_image"]

    def _add_ea_magnet_names(self):
        """
        Some old recordings did not include the magnet names in `info`. Those should all
        have been recorded on the EA. So if no magnet names are included, add the EA
        magnet names.
        """
        for info in self.infos:
            if "magnet_names" not in info:
                info["magnet_names"] = [
                    "AREAMQZM1",
                    "AREAMQZM2",
                    "AREAMCVM1",
                    "AREAMQZM3",
                    "AREAMCHM1",
                ]

    def sum_of_normalized_magnet_changes(self) -> float:
        """
        Compute the sum of the normalized changes in magnet settings over the episode.
        """
        magnets = self.magnet_history()
        normalized_magnets = magnets / np.array([30, 30, 6.1782e-3, 30, 6.1782e-3])
        changes = np.abs(np.diff(normalized_magnets, axis=0))
        return np.sum(changes)
