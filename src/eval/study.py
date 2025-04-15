from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns

from . import Episode

plt.style.use(["science", "nature", "no-latex"])


class Study:
    """
    A study comprising multiple optimisation runs.
    """

    def __init__(self, episodes: list[Episode], name: Optional[str] = None) -> None:
        assert len(episodes) > 0, "No episodes passed to study at initialisation."

        self.episodes = episodes
        self.name = name

    @classmethod
    def load(
        cls,
        data_dir: Union[Path, str],
        runs: Union[str, list[str]] = "*problem_*",
        name: Optional[str] = None,
        drop_screen_images: bool = False,
        use_problem_index: bool = True,
    ) -> Study:
        """
        Loads all episode pickle files from an evaluation firectory. Expects
        `problem_xxx` directories, each of which has a `recorded_episdoe_1.pkl file in
        it.
        """
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        assert data_dir.is_dir(), f"Data directory {data_dir} does not exist."

        run_paths = (
            data_dir.glob(runs)
            if isinstance(runs, str)
            else [data_dir / run for run in runs]
        )
        paths = [p / "recorded_episode_1.pkl" for p in sorted(run_paths)]
        assert len(paths) > 0, f"No episodes found in {data_dir}."
        episodes = [
            Episode.load(
                p,
                use_problem_index=use_problem_index,
                drop_screen_images=drop_screen_images,
            )
            for p in paths
        ]

        return Study(episodes, name=name)

    def __len__(self) -> int:
        """A study's length is the number of episodes in it."""
        return len(self.episodes)

    def head(self, n: int) -> Study:
        """Return study with `n` first steps from all episodes in this study."""
        return Study(
            episodes=[episode.head(n) for episode in self.episodes],
            name=f"{self.name} - head",
        )

    def tail(self, n: int) -> Study:
        """Return study with `n` last steps from all episodes in this study."""
        return Study(
            episodes=[episode.tail(n) for episode in self.episodes],
            name=f"{self.name} - tail",
        )

    def problem_intersection(self, other: Study, rename: bool = False) -> Study:
        """
        Return a new study from the intersection of problems with the `other` study.
        """
        my_problems = set(self.problem_indicies())
        other_problems = set(other.problem_indicies())

        episodes = [
            self.get_episodes_by_problem(problem)[0]
            for problem in my_problems.intersection(other_problems)
        ]

        return Study(
            episodes=episodes,
            name=f"{self.name} ∩ {other.name}" if rename else self.name,
        )

    def median_best_mae(self) -> float:
        """Compute median of best MAEs seen until the very end of the episodes."""
        return np.median([episode.best_mae() for episode in self.episodes])

    def mean_best_mae(self) -> float:
        """Compute mean of best MAEs seen until the very end of the episodes."""
        return np.mean([episode.best_mae() for episode in self.episodes])

    def std_best_mae(self) -> float:
        """
        Compute standard deviation of best MAEs seen until the very end of the episodes.
        """
        return np.std([episode.best_mae() for episode in self.episodes])

    def median_final_mae(self) -> float:
        """
        Median of the final MAE that the algorithm stopped at (without returning to best
        seen).
        """
        return np.median([episode.final_mae() for episode in self.episodes])

    def mean_final_mae(self) -> float:
        """
        Mean of the final MAE that the algorithm stopped at (without returning to best
        seen).
        """
        return np.mean([episode.final_mae() for episode in self.episodes])

    def std_final_mae(self) -> float:
        """
        Standard deviation of the final MAE that the algorithm stopped at (without
        returning to best seen).
        """
        return np.std([episode.final_mae() for episode in self.episodes])

    def median_steps_to_convergence(
        self, threshold: float = 20e-6, max_steps: Optional[int] = None
    ) -> Optional[float]:
        """
        Median number of steps until best seen MAE no longer improves by more than
        `threshold`. If `max_steps` is given, only consider episodes that converged in
        less than `max_steps`. Returns `None` if no runs got there in less than
        `max_steps`.
        """
        steps = [episode.steps_to_convergence(threshold) for episode in self.episodes]

        if max_steps:
            steps = np.array(steps)
            steps = steps[steps < max_steps]

        return np.median(steps) if len(steps) > 0 else None

    def mean_steps_to_convergence(
        self, threshold: float = 20e-6, max_steps: Optional[int] = None
    ) -> Optional[float]:
        """
        Mean number of steps until best seen MAE no longer improves by more than
        `threshold`. If `max_steps` is given, only consider episodes that converged in
        less than `max_steps`. Returns `None` if no runs got there in less than
        `max_steps`.
        """
        steps = [episode.steps_to_convergence(threshold) for episode in self.episodes]

        if max_steps:
            steps = np.array(steps)
            steps = steps[steps < max_steps]

        return np.mean(steps) if len(steps) > 0 else None

    def std_steps_to_convergence(
        self, threshold: float = 20e-6, max_steps: Optional[int] = None
    ) -> Optional[float]:
        """
        Standard deviation of number of steps until best seen MAE no longer improves by
        more than `threshold`. If `max_steps` is given, only consider episodes that
        converged in less than `max_steps`. Returns `None` if no runs got there in less
        than `max_steps`.
        """
        steps = [episode.steps_to_convergence(threshold) for episode in self.episodes]

        if max_steps:
            steps = np.array(steps)
            steps = steps[steps < max_steps]

        return np.std(steps) if len(steps) > 0 else None

    def median_steps_to_threshold(
        self,
        threshold: float = 20e-6,
        max_steps: Optional[int] = None,
        use_min_mae: bool = True,
        allow_lowest_as_target: bool = False,
    ) -> Optional[float]:
        """
        Median number of steps until best seen MAE drops below (resolution) `threshold`.
        If `max_steps` is given, only consider episodes that got below threshold in less
        than `max_steps`. Returns `None` if no runs got there in less than `max_steps`.
        """
        steps = [
            episode.steps_to_threshold(
                threshold=threshold,
                use_min_mae=use_min_mae,
                allow_lowest_as_target=allow_lowest_as_target,
            )
            for episode in self.episodes
            if episode.best_mae() < threshold
        ]

        if max_steps:
            steps = np.array(steps)
            steps = steps[steps < max_steps]

        return np.median(steps) if len(steps) > 0 else None

    def mean_steps_to_threshold(
        self,
        threshold: float = 20e-6,
        max_steps: Optional[int] = None,
        use_min_mae: bool = True,
        allow_lowest_as_target: bool = False,
    ) -> Optional[float]:
        """
        Mean number of steps until best seen MAE drops below (resolution) `threshold`.
        If `max_steps` is given, only consider episodes that got below threshold in less
        than `max_steps`. Returns `None` if no runs got there in less than `max_steps`.
        """
        steps = [
            episode.steps_to_threshold(
                threshold=threshold,
                use_min_mae=use_min_mae,
                allow_lowest_as_target=allow_lowest_as_target,
            )
            for episode in self.episodes
            if episode.best_mae() < threshold
        ]

        if max_steps:
            steps = np.array(steps)
            steps = steps[steps < max_steps]

        return np.mean(steps) if len(steps) > 0 else None

    def std_steps_to_threshold(
        self,
        threshold: float = 20e-6,
        max_steps: Optional[int] = None,
        use_min_mae: bool = True,
        allow_lowest_as_target: bool = False,
    ) -> Optional[float]:
        """
        Standard deviation of number of steps until best seen MAE drops below
        (resolution) `threshold`. If `max_steps` is given, only consider episodes that
        got below threshold in less than `max_steps`. Returns `None` if no runs got
        there in less than `max_steps`.
        """
        steps = [
            episode.steps_to_threshold(
                threshold=threshold,
                use_min_mae=use_min_mae,
                allow_lowest_as_target=allow_lowest_as_target,
            )
            for episode in self.episodes
            if episode.best_mae() < threshold
        ]

        if max_steps:
            steps = np.array(steps)
            steps = steps[steps < max_steps]

        return np.std(steps) if len(steps) > 0 else None

    def rmse(
        self, max_steps: Optional[int] = None, use_best_beam: bool = False
    ) -> float:
        """
        RMSE over all samples in all episode in the study and over all beam parameters
        as used in https://www.nature.com/articles/s41586-021-04301-9.

        This particular implementation is similar to the Frobenius norm.
        """
        if use_best_beam:
            beams = np.stack(
                [episode.best_beam_history()[:max_steps] for episode in self.episodes]
            ).reshape(-1, 4)
        else:
            beams = np.stack(
                [
                    obs["beam"]
                    for episode in self.episodes
                    for obs in episode.observations[:max_steps]
                ]
            )
        targets = np.stack(
            [
                obs["target"]
                for episode in self.episodes
                for obs in episode.observations[:max_steps]
            ]
        )
        rmse = np.sqrt(np.mean(np.square(targets - beams)))
        return rmse

    def median_rmse(self, use_best_beam: bool = False) -> float:
        """
        Median RMSE over all samples in all episode in the study and over all beam
        parameters as used in https://www.nature.com/articles/s41586-021-04301-9.
        """
        episode_rmses = [
            episode.rmse(use_best_beam=use_best_beam) for episode in self.episodes
        ]
        return np.median(episode_rmses)

    def mean_rmse(self, use_best_beam: bool = False) -> float:
        """
        Mean RMSE over all samples in all episode in the study and over all beam
        parameters as used in https://www.nature.com/articles/s41586-021-04301-9.
        """
        episode_rmses = [
            episode.rmse(use_best_beam=use_best_beam) for episode in self.episodes
        ]
        return np.mean(episode_rmses)

    def std_rmse(self, use_best_beam: bool = False) -> float:
        """
        Standard deviation of RMSE over all samples in all episode in the study and over
        all beam parameters as used in
        https://www.nature.com/articles/s41586-021-04301-9.
        """
        episode_rmses = [
            episode.rmse(use_best_beam=use_best_beam) for episode in self.episodes
        ]
        return np.std(episode_rmses)

    def median_mae_improvement(
        self, use_min_mae: bool = False, include_before_reset: bool = False
    ) -> float:
        """
        Median improvement in MAE from the first to the last step of the episode.
        Positive values indicate improvement, negative values indicate worsening.
        """
        improvements = [
            episode.mae_improvement(
                use_min_mae=use_min_mae, include_before_reset=include_before_reset
            )
            for episode in self.episodes
        ]
        return np.median(improvements)

    def mean_mae_improvement(
        self, use_min_mae: bool = False, include_before_reset: bool = False
    ) -> float:
        """
        Mean improvement in MAE from the first to the last step of the episode.
        Positive values indicate improvement, negative values indicate worsening.
        """
        improvements = [
            episode.mae_improvement(
                use_min_mae=use_min_mae, include_before_reset=include_before_reset
            )
            for episode in self.episodes
        ]
        return np.mean(improvements)

    def std_mae_improvement(
        self, use_min_mae: bool = False, include_before_reset: bool = False
    ) -> float:
        """
        Standard deviation of improvement in MAE from the first to the last step of the
        episode. Positive values indicate improvement, negative values indicate
        worsening.
        """
        improvements = [
            episode.mae_improvement(
                use_min_mae=use_min_mae, include_before_reset=include_before_reset
            )
            for episode in self.episodes
        ]
        return np.std(improvements)

    def median_normalized_mae_improvement(
        self, use_min_mae: bool = False, include_before_reset: bool = False
    ) -> float:
        """
        Median improvement in MAE from the first to the last step of the episode,
        normalized to the initial MAE. Positive values indicate improvement, negative
        values indicate worsening.
        """
        improvements = [
            episode.normalized_mae_improvement(
                use_min_mae=use_min_mae, include_before_reset=include_before_reset
            )
            for episode in self.episodes
        ]
        return np.median(improvements)

    def mean_normalized_mae_improvement(
        self, use_min_mae: bool = False, include_before_reset: bool = False
    ) -> float:
        """
        Mean improvement in MAE from the first to the last step of the episode,
        normalized to the initial MAE. Positive values indicate improvement, negative
        values indicate worsening.
        """
        improvements = [
            episode.normalized_mae_improvement(
                use_min_mae=use_min_mae, include_before_reset=include_before_reset
            )
            for episode in self.episodes
        ]
        return np.mean(improvements)

    def std_normalized_mae_improvement(
        self, use_min_mae: bool = False, include_before_reset: bool = False
    ) -> float:
        """
        Standard deviation of improvement in MAE from the first to the last step of the
        episode, normalized to the initial MAE. Positive values indicate improvement,
        negative values indicate worsening.
        """
        improvements = [
            episode.normalized_mae_improvement(
                use_min_mae=use_min_mae, include_before_reset=include_before_reset
            )
            for episode in self.episodes
        ]
        return np.std(improvements)

    def problem_indicies(self) -> list[int]:
        """
        Return unsorted list of problem indicies in this study. `None` is returned for
        problems that do not have a problem index.
        """
        return [episode.problem_index for episode in self.episodes]

    def get_episodes_by_problem(self, i: int) -> list[Episode]:
        """Get all episodes in this study that have problem index `i`."""
        return [episode for episode in self.episodes if episode.problem_index == i]

    def all_episodes_have_problem_index(self) -> bool:
        """
        Check if all episodes in this study have a problem index associated with them.
        """
        return all(hasattr(episode, "problem_index") for episode in self.episodes)

    def are_problems_unique(self) -> bool:
        """Check if there is at most one of each problem (index)."""
        idxs = self.problem_indicies()
        return len(idxs) == len(set(idxs))

    def plot_best_mae_over_problem(self) -> None:
        """
        Plot the best MAE achieved for each problem to see if certain problems stand
        out.
        """
        assert (
            self.all_episodes_have_problem_index()
        ), "At least on episode in this study does not have a problem index."
        assert self.are_problems_unique(), "There are duplicate problems in this study."

        sorted_problems = sorted(self.problem_indicies())
        sorted_episodes = [
            self.get_episodes_by_problem(problem)[0] for problem in sorted_problems
        ]
        best_maes = [episode.best_mae() for episode in sorted_episodes]

        plt.figure(figsize=(5, 3))
        plt.bar(sorted_problems, best_maes, label=self.name)
        plt.legend()
        plt.xlabel("Problem Index")
        plt.ylabel("Best MAE")
        plt.show()

    def plot_target_beam_size_mae_correlation(self) -> None:
        """Plot best MAEs over mean target beam size to see possible correlation."""

        best_mae = [episode.best_mae() for episode in self.episodes]
        target_sizes = [episode.target_size() for episode in self.episodes]

        plt.figure(figsize=(5, 3))
        plt.scatter(target_sizes, best_mae, s=3, label=self.name)
        plt.legend()
        plt.xlabel("Mean beam size x/y")
        plt.ylabel("Best MAE")
        plt.show()

    def plot_best_return_deviation_box(
        self, print_results: bool = True, save_path: Optional[str] = None
    ) -> None:
        """
        Plot a boxplot showing how far the MAE in the final return step differed from
        the MAE seen the first time the optimal magnets were set. This should show
        effects of hysteresis (and simular effects).
        """
        best = [episode.best_mae() for episode in self.episodes]
        final = [episode.final_mae() for episode in self.episodes]
        deviations = np.abs(np.array(best) - np.array(final))

        if print_results:
            print(f"Median deviation = {np.median(deviations)}")
            print(f"Max deviation = {np.max(deviations)}")

        plt.figure(figsize=(5, 2))
        plt.title("Deviation when returning to best")
        sns.boxplot(x=deviations, y=["Deviation"] * len(deviations))
        plt.gca().set_axisbelow(True)
        plt.xlabel("MAE")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def average_inference_times(self):
        """
        Return average time it took to infer the next action/magnet setting throughout
        the study.
        """
        first_inferences = [
            episode.step_start_times[0] - episode.t_start
            for episode in self.episodes
            if len(episode) > 1
        ]
        other_inferences = [
            t2 - t1
            for episode in self.episodes
            for t1, t2 in zip(episode.step_end_times[:-1], episode.step_start_times[1:])
            if len(episode) > 1
        ]
        return np.mean(first_inferences + other_inferences)

    def std_inference_times(self):
        """
        Return the standard deviation of the times it took to infer the next
        action/magnet setting throughout the study.
        """
        first_inferences = [
            episode.step_start_times[0] - episode.t_start for episode in self.episodes
        ]
        other_inferences = [
            t2 - t1
            for episode in self.episodes
            for t1, t2 in zip(episode.step_end_times[:-1], episode.step_start_times[1:])
        ]

        # Trick needed because numpy won't compute std of datetime.timedeltas
        floats_seconds = [
            td.total_seconds() for td in first_inferences + other_inferences
        ]
        std_dev_seconds = np.std(floats_seconds)
        std_dev_timedelta = timedelta(seconds=std_dev_seconds)

        return std_dev_timedelta

    def plot_correlations_to_mae(self):
        """
        Plot a MAE in relation to a number of different variables to investigate
        possible correlations with optimiser performance.
        """
        final_maes = [episode.final_mae() for episode in self.episodes]
        target_beams = np.array(
            [episode.observations[0]["target"] for episode in self.episodes]
        )
        initial_beams = np.array(
            [episode.observations[0]["beam"] for episode in self.episodes]
        )
        if all(
            "misalignments" in episode.observations[0]
            and "incoming" in episode.observations[0]
            for episode in self.episodes
        ):
            rows = 23
            has_misalignments = True
            has_incoming_beams = True
            misalignments = np.array(
                [episode.observations[0]["misalignments"] for episode in self.episodes]
            )
            incoming_beams = np.array(
                [episode.observations[0]["incoming"] for episode in self.episodes]
            )[:, [1, 5, 3, 7]]
        else:
            rows = 11
            has_misalignments = False
            has_incoming_beams = False

        height = 2.5 * rows

        plt.figure(figsize=(15, height))

        for i in range(4):
            plt.subplot(rows, 4, 0 + i + 1)
            sns.regplot(x=target_beams[:, i], y=final_maes)
            plt.xlabel(f"Target beam {i}")
            plt.ylabel("MAE")
        for i in range(4):
            plt.subplot(rows, 4, 4 + i + 1)
            sns.regplot(x=np.abs(target_beams[:, i]), y=final_maes)
            plt.xlabel(f"|Target beam {i}|")
            plt.ylabel("MAE")
        plt.subplot(rows, 4, 8 + 1)
        sns.regplot(x=np.sum(np.abs(target_beams), axis=1), y=final_maes)
        plt.xlabel(r"$\sum$ target beams")
        plt.ylabel("MAE")
        plt.subplot(rows, 4, 12 + 1)
        sns.regplot(x=np.sum(np.abs(target_beams[:, [1, 3]]), axis=1), y=final_maes)
        plt.xlabel(r"$\sum$ target beam sizes")
        plt.ylabel("MAE")
        for i in range(4):
            plt.subplot(rows, 4, 16 + i + 1)
            sns.regplot(x=initial_beams[:, i], y=final_maes)
            plt.xlabel(f"Initial beam {i}")
            plt.ylabel("MAE")
        for i in range(4):
            plt.subplot(rows, 4, 20 + i + 1)
            sns.regplot(x=np.abs(initial_beams[:, i]), y=final_maes)
            plt.xlabel(f"|Initial beam {i}|")
            plt.ylabel("MAE")
        plt.subplot(rows, 4, 24 + 1)
        sns.regplot(x=np.sum(np.abs(initial_beams), axis=1), y=final_maes)
        plt.xlabel(r"$\sum$ Target beam")
        plt.ylabel("MAE")
        plt.subplot(rows, 4, 28 + 1)
        sns.regplot(x=np.sum(np.abs(initial_beams[:, [1, 3]]), axis=1), y=final_maes)
        plt.xlabel(r"$\sum$ Target beam size")
        plt.ylabel("MAE")
        for i in range(4):
            plt.subplot(rows, 4, 32 + i + 1)
            sns.regplot(x=target_beams[:, i] - initial_beams[:, i], y=final_maes)
            plt.xlabel(f"Target beam - initial beam {i}")
            plt.ylabel("MAE")
        for i in range(4):
            plt.subplot(rows, 4, 36 + i + 1)
            sns.regplot(
                x=np.abs(target_beams[:, i] - initial_beams[:, i]), y=final_maes
            )
            plt.xlabel(f"|Target beam - initial beam {i}|")
            plt.ylabel("MAE")
        plt.subplot(rows, 4, 40 + 1)
        sns.regplot(
            x=np.sum(np.abs(target_beams - initial_beams), axis=1), y=final_maes
        )
        plt.xlabel(r"$\sum$ |Target beam - initial beam|")
        plt.ylabel("MAE")

        if has_misalignments:
            for i in range(8):
                plt.subplot(rows, 4, 44 + i + 1)
                sns.regplot(x=misalignments[:, i], y=final_maes)
                plt.xlabel(f"Misalignment {i}")
                plt.ylabel("MAE")
            for i in range(8):
                plt.subplot(rows, 4, 52 + i + 1)
                sns.regplot(x=np.abs(misalignments[:, i]), y=final_maes)
                plt.xlabel(f"|Misalignment {i}|")
                plt.ylabel("MAE")
            plt.subplot(rows, 4, 60 + 1)
            sns.regplot(x=np.sum(np.abs(misalignments), axis=1), y=final_maes)
            plt.xlabel(r"$\sum$ misalignments")
            plt.ylabel("MAE")

        if has_incoming_beams:
            for i in range(4):
                plt.subplot(rows, 4, 64 + i + 1)
                sns.regplot(x=incoming_beams[:, i], y=final_maes)
                plt.xlabel(f"Incoming beam {i}")
                plt.ylabel("MAE")
            for i in range(4):
                plt.subplot(rows, 4, 68 + i + 1)
                sns.regplot(x=np.abs(incoming_beams[:, i]), y=final_maes)
                plt.xlabel(f"|Incoming beam {i}|")
                plt.ylabel("MAE")
            plt.subplot(rows, 4, 72 + 1)
            sns.regplot(x=np.sum(np.abs(incoming_beams), axis=1), y=final_maes)
            plt.xlabel(r"$\sum$ |Incoming beam|")
            plt.ylabel("MAE")
            plt.subplot(rows, 4, 76 + 1)
            sns.regplot(
                x=np.sum(np.abs(incoming_beams[:, [1, 3]]), axis=1), y=final_maes
            )
            plt.xlabel(r"$\sum$ |Incoming beam size|")
            plt.ylabel("MAE")
            for i in range(4):
                plt.subplot(rows, 4, 80 + i + 1)
                sns.regplot(x=target_beams[:, i] - incoming_beams[:, i], y=final_maes)
                plt.xlabel(f"Target beam - incoming beam {i}")
                plt.ylabel("MAE")
            for i in range(4):
                plt.subplot(rows, 4, 84 + i + 1)
                sns.regplot(
                    x=np.abs(target_beams[:, i] - incoming_beams[:, i]), y=final_maes
                )
                plt.xlabel(f"|Target beam - incoming beam {i}|")
                plt.ylabel("MAE")
            plt.subplot(rows, 4, 88 + 1)
            sns.regplot(
                x=np.sum(np.abs(target_beams - incoming_beams), axis=1), y=final_maes
            )
            plt.xlabel(r"$\sum$ |Target beam - incoming beam|")
            plt.ylabel("MAE")

        plt.tight_layout()
        plt.show()

    def proportion_reached_target(
        self, threshold: float = 2e-5, max_steps: int = 50, use_min_mae: bool = True
    ) -> float:
        """
        Return the proportion of episodes that reached a MAE below `threshold` within
        `max_steps`.
        """
        num_reached = sum(
            np.array(
                [
                    episode.steps_to_threshold(
                        threshold=threshold, use_min_mae=use_min_mae
                    )
                    for episode in self.episodes
                ]
            )
            < max_steps
        )
        proportion = num_reached / len(self)
        return proportion

    def proportion_converged(
        self, threshold: float = 2e-5, max_steps: int = 50, use_min_mae: bool = True
    ) -> float:
        """
        Return the proportion of episodes that converged (improved by less than
        `threshold`) within `max_steps`.
        """
        num_converged = sum(
            np.array(
                [
                    episode.steps_to_convergence(threshold, use_min_mae=use_min_mae)
                    for episode in self.episodes
                ]
            )
            < max_steps
        )
        proportion = num_converged / len(self)
        return proportion

    def mean_accumulated_mae(self, use_min_mae: bool = False) -> float:
        """
        Return the mean accumulated MAE over all episodes in the study.
        """
        return np.mean(
            [episode.accumulated_mae(use_min_mae) for episode in self.episodes]
        )

    def mean_normalized_accumulated_mae(self, use_min_mae: bool = False) -> float:
        """
        Return the mean normalized accumulated MAE over all episodes in the study.
        """
        return np.mean(
            [
                episode.normalized_accumulated_mae(use_min_mae)
                for episode in self.episodes
            ]
        )

    def median_sum_of_normalized_magnet_changes(self) -> float:
        """
        Return the median sum of normalized magnet changes over all episodes in the
        study.
        """
        return np.median(
            [episode.sum_of_normalized_magnet_changes() for episode in self.episodes]
        )

    def evaluate_challenge(self) -> None:
        """Produce a CSV file for the Kaggle challenge and output evaluation results."""
        # Prodcue Kaggle CSV
        final_mae = [episode.final_mae() for episode in self.episodes]
        steps_to_convergence = [
            episode.steps_to_convergence(4e-5) for episode in self.episodes
        ]
        sum_of_normalized_magnet_changes = [
            episode.sum_of_normalized_magnet_changes() for episode in self.episodes
        ]
        df = pd.DataFrame(
            {
                "initial_mae": [episode.maes()[0] for episode in self.episodes],
                "final_mae": final_mae,
                "steps_to_convergence": steps_to_convergence,
                "sum_of_normalized_magnet_changes": sum_of_normalized_magnet_changes,
                "i01": [
                    episode.infos[0]["backend_info"]["incoming_beam"][1]
                    for episode in self.episodes
                ],
                "i02": [
                    episode.infos[0]["backend_info"]["incoming_beam"][2]
                    for episode in self.episodes
                ],
                "i03": [
                    episode.infos[0]["backend_info"]["incoming_beam"][3]
                    for episode in self.episodes
                ],
                "i04": [
                    episode.infos[0]["backend_info"]["incoming_beam"][4]
                    for episode in self.episodes
                ],
                "m02": [
                    episode.infos[0]["backend_info"]["misalignments"][2]
                    for episode in self.episodes
                ],
            }
        )
        # Validate steps to convergence (should be max steps if final MAE is not less
        # than final MAE)
        df["steps_to_convergence"] = df.apply(
            lambda row: (
                row["steps_to_convergence"]
                if row["final_mae"] < row["initial_mae"]
                else 150
            ),
            axis=1,
        )
        Path("data/csvs").mkdir(exist_ok=True)
        if Path(f"data/csvs/{self.name}.csv").exists():
            print(f"Overwriting existing file data/csvs/{self.name}.csv")
        df.to_csv(f"data/csvs/{self.name}.csv", index_label="id")

        median_final_mae = np.median(df["final_mae"])
        median_steps_to_convergence = np.median(df["steps_to_convergence"])
        median_sum_of_normalized_magnet_changes = np.median(
            df["sum_of_normalized_magnet_changes"]
        )

        # Output evaluation results
        print(f"Final MAE: {median_final_mae * 1e6:.0f} μm")
        print(f"Steps to convergence: {median_steps_to_convergence:.1f}")
        print(f"Sum of magnet changes: {median_sum_of_normalized_magnet_changes:.2f} ")

        score = (
            3 * median_final_mae * 250
            + 0.5 * median_steps_to_convergence / 150
            + 0.5 * median_sum_of_normalized_magnet_changes / (5 * 150)
        )
        print("--------------------")
        print(f"Score: {score.mean():.4f}")
