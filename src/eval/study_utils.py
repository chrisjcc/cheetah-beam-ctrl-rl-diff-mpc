from typing import Any, Literal, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns

from . import Study

plt.style.use(["science", "nature", "no-latex"])


def number_of_better_final_beams(study_1: Study, study_2: Study) -> int:
    """
    Computer the number of times that the best MAE of a run in `study_1` is better than
    the best MAE of the same run in `study_2`.
    """
    assert study_1.are_problems_unique(), "The problems in study 1 are note unique."
    assert study_2.are_problems_unique(), "The problems in study 2 are note unique."

    study_1_idxs = sorted(study_1.problem_indicies())
    study_2_idxs = sorted(study_2.problem_indicies())
    assert study_1_idxs == study_2_idxs, "The studies do not cover the same problems."

    problem_idxs = study_1_idxs
    best_maes_1 = [
        study_1.get_episodes_by_problem(i)[0].best_mae() for i in problem_idxs
    ]
    best_maes_2 = [
        study_2.get_episodes_by_problem(i)[0].best_mae() for i in problem_idxs
    ]

    diff = np.array(best_maes_1) - np.array(best_maes_2)

    return sum(diff < 0)


def problem_aligned(studies: list[Study]) -> list[Study]:
    """
    Intersect the problems of all `studies` such that the studies in the returned list
    all cover exactly the same problems.
    """
    # Find the smallest intersection of problem indicies
    intersected = set(studies[0].problem_indicies())
    for study in studies:
        intersected = intersected.intersection(set(study.problem_indicies()))

    new_studies = []
    for study in studies:
        intersected_study = Study(
            episodes=[study.get_episodes_by_problem(i)[0] for i in intersected],
            name=study.name,
        )
        new_studies.append(intersected_study)

    return new_studies


def plot_best_mae_box(
    studies: list[Study],
    palette: Optional[Any] = None,
    saturation: float = 0.75,
    figsize: tuple[float, float] = (6, 3),
    ax: Optional[matplotlib.axes.Axes] = None,
    save_path: Optional[str] = None,
) -> None:
    """Box plot of best MAEs seen until the very end of the episodes."""
    combined_best_maes = []
    combined_names = []
    for study in studies:
        best_maes = [episode.best_mae() for episode in study.episodes]

        names = [study.name] * len(best_maes)

        combined_best_maes += best_maes
        combined_names += names

    combined_best_maes = np.array(combined_best_maes) * 1e3  # Convert to millimeters

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title("Best MAEs")
    sns.boxplot(
        x=combined_best_maes,
        y=combined_names,
        hue=combined_names,
        palette=palette,
        saturation=saturation,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("MAE (mm)")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="minor", left=False, right=False)

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        fig.savefig(save_path)


def plot_best_mae_diff_over_problem(
    study_1: Study,
    study_2: Study,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the differences of the best MAE achieved for each problem to see if certain
    problems stand out.
    """
    assert study_1.are_problems_unique(), "The problems in study 1 are note unique."
    assert study_2.are_problems_unique(), "The problems in study 2 are note unique."

    study_1_idxs = sorted(study_1.problem_indicies())
    study_2_idxs = sorted(study_2.problem_indicies())
    assert study_1_idxs == study_2_idxs, "The studies do not cover the same problems."

    problem_idxs = study_1_idxs
    best_maes_1 = [
        study_1.get_episodes_by_problem(i)[0].best_mae() for i in problem_idxs
    ]
    best_maes_2 = [
        study_2.get_episodes_by_problem(i)[0].best_mae() for i in problem_idxs
    ]

    diff = np.array(best_maes_1) - np.array(best_maes_2)

    plt.figure(figsize=(5, 3))
    plt.bar(problem_idxs, diff, label=f"{study_1.name} vs. {study_2.name}")
    plt.legend()
    plt.xlabel("Problem Index")
    plt.ylabel("Best MAE")

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_best_mae_over_time(
    studies: list[Study],
    threshold: Optional[float] = None,
    logarithmic: bool = False,
    legend: bool = True,
    title: Optional[str] = "Mean Best MAE Over Time",
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: tuple[float, float] = (5, 3),
    study_name_str: str = "Study",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot mean best seen MAE over all episdoes over time. Optionally display a
    `threshold` line to mark measurement limit. Set `logarithmic` to `True` to log scale
    the y-axis.
    """
    dfs = []
    for study in studies:
        ds = [
            {
                "MAE (m)": episode.min_maes(),
                "Step": range(len(episode)),
                "Problem Index": episode.problem_index,
                study_name_str: study.name,
            }
            for episode in study.episodes
        ]
        df = pd.concat(pd.DataFrame(d) for d in ds)

        dfs.append(df)

    combined_df = pd.concat(dfs)

    # Convert unit to mm
    combined_df["MAE (mm)"] = combined_df["MAE (m)"] * 1e3

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if threshold is not None:
        ax.axhline(threshold, ls="--", color="lightsteelblue", label="Threshold")
    sns.lineplot(
        x="Step",
        y="MAE (mm)",
        hue=study_name_str,
        data=combined_df,
        legend=legend,
        ax=ax,
    )
    ax.set_ylabel("MAE (mm)")
    ax.set_title(title)
    ax.set_xlim(0, None)
    if logarithmic:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, None)

    ax.set_axisbelow(True)

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        plt.savefig(save_path)


def plot_final_mae_box(
    studies: list[Study],
    title: Optional[str] = "Final MAEs",
    palette: Optional[Any] = None,
    saturation: float = 0.75,
    figsize: tuple[float, float] = (5, 3),
    ax: Optional[matplotlib.axes.Axes] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Box plot of the final MAE that the algorithm stopped at (without returning to best
    seen).
    """
    combined_final_maes = []
    combined_names = []
    for study in studies:
        final_maes = [episode.final_mae() for episode in study.episodes]

        names = [study.name] * len(final_maes)

        combined_final_maes += final_maes
        combined_names += names

    combined_final_maes = np.array(combined_final_maes) * 1e3  # Convert to millimeters

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title)
    sns.boxplot(
        x=combined_final_maes,
        y=combined_names,
        hue=combined_names,
        palette=palette,
        saturation=saturation,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("MAE (mm)")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="minor", left=False, right=False)

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        fig.savefig(save_path)


def plot_mae_over_time(
    studies: list[Study],
    threshold: Optional[float] = None,
    logarithmic: bool = False,
    legend: bool = True,
    title: Optional[str] = "Mean MAE Over Time",
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: tuple[float, float] = (5, 3),
    study_name_str: str = "Study",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot mean MAE of over episodes over time. Optionally display a `threshold` line to
    mark measurement limit. Set `logarithmic` to `True` to log scale the y-axis.
    """
    dfs = []
    for study in studies:
        ds = [
            {
                "MAE (m)": episode.maes(),
                "Step": range(len(episode)),
                "Problem Index": episode.problem_index,
                study_name_str: study.name,
            }
            for episode in study.episodes
        ]
        df = pd.concat(pd.DataFrame(d) for d in ds)

        dfs.append(df)

    combined_df = pd.concat(dfs)

    # Convert unit to mm
    combined_df["MAE (mm)"] = combined_df["MAE (m)"] * 1e3

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if threshold is not None:
        ax.axhline(threshold, ls="--", color="lightsteelblue", label="Threshold")
    sns.lineplot(
        x="Step",
        y="MAE (mm)",
        hue=study_name_str,
        data=combined_df,
        legend=legend,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("MAE (mm)")
    ax.set_xlim(0, None)
    if logarithmic:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, None)

    ax.set_axisbelow(True)

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        plt.savefig(save_path)


def plot_steps_to_convergence_box(
    studies: list[Study],
    threshold: float = 20e-6,
    title: Optional[str] = "Steps to convergence",
    palette: Optional[Any] = None,
    saturation: float = 0.75,
    figsize: tuple[float, float] = (5, 3),
    ax: Optional[matplotlib.axes.Axes] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Box plot number of steps until best seen MAE no longer improves by more than
    `threshold`.
    """
    combined_steps = []
    combined_names = []
    for study in studies:
        steps = [episode.steps_to_convergence(threshold) for episode in study.episodes]
        names = [study.name] * len(steps)

        combined_steps += steps
        combined_names += names

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title)
    sns.boxplot(
        x=combined_steps,
        y=combined_names,
        hue=combined_names,
        palette=palette,
        saturation=saturation,
        ax=ax,
    )
    ax.set_axisbelow(True)
    ax.set_xlabel("No. of steps")
    ax.tick_params(axis="y", which="minor", left=False, right=False)

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        fig.savefig(save_path)


def plot_steps_to_threshold_box(
    studies: list[Study],
    threshold: float = 20e-6,
    title: Optional[str] = "Steps to MAE below threshold",
    palette: Optional[Any] = None,
    figsize: tuple[float, float] = (5, 3),
    ax: Optional[matplotlib.axes.Axes] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Box plot number of steps until best seen MAE drops below (resolution) `threshold`.
    """
    combined_steps = []
    combined_names = []
    for study in studies:
        steps = [episode.steps_to_threshold(threshold) for episode in study.episodes]
        names = [study.name] * len(steps)

        combined_steps += steps
        combined_names += names

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title)
    sns.boxplot(
        x=combined_steps, y=combined_names, hue=combined_names, palette=palette, ax=ax
    )
    ax.set_axisbelow(True)
    ax.set_xlabel("No. of steps")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="minor", left=False, right=False)

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        fig.savefig(save_path)


def plot_final_beam_parameter_error_box(
    studies: list[Study],
    name: Literal["mu_x", "sigma_x", "mu_y", "sigma_y"],
    figsize: tuple[float, float] = (6, 3),
    ax: Optional[matplotlib.axes.Axes] = None,
    save_path: Optional[str] = None,
) -> None:
    """Box plot of final beam parameter at end of the episodes."""
    parameter_index = {"mu_x": 0, "sigma_x": 1, "mu_y": 2, "sigma_y": 3}[name]

    combined_error = []
    combined_names = []
    for study in studies:
        final_errors = [episode.error_history()[-1] for episode in study.episodes]
        selected_errors = [error[parameter_index] for error in final_errors]

        names = [study.name] * len(selected_errors)

        combined_error += selected_errors
        combined_names += names

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(f"Final error for beam parameter {name}")
    sns.boxplot(x=combined_error, y=combined_names, hue=combined_names, ax=ax)
    ax.set_xscale("log")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="minor", left=False, right=False)

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        fig.savefig(save_path)


def plot_best_beam_parameter_error_box(
    studies: list[Study],
    name: Literal["mu_x", "sigma_x", "mu_y", "sigma_y"],
    figsize: tuple[float, float] = (6, 3),
    ax: Optional[matplotlib.axes.Axes] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Box plot of best value seen for the selected beam parameter seen until the very end
    of the episodes.
    """
    combined_error = []
    combined_names = []
    for study in studies:
        best_errors = [
            episode.best_beam_parameter_error(name) for episode in study.episodes
        ]

        names = [study.name] * len(best_errors)

        combined_error += best_errors
        combined_names += names

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(f"Best error for beam parameter {name}")
    sns.boxplot(x=combined_error, y=combined_names, hue=combined_names)
    ax.set_xscale("log")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="minor", left=False, right=False)

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        fig.savefig(save_path)


def plot_rmse_box(
    studies: list[Study],
    title: Optional[str] = "RMSEs",
    palette: Optional[Any] = None,
    saturation: float = 0.75,
    figsize: tuple[float, float] = (5, 3),
    ax: Optional[matplotlib.axes.Axes] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Box plot of the RMSEs achieved in each study as the mean RMSE for each episode,
    where the RMSE is computed over the entire episode temporally.
    """
    combined_rmses = []
    combined_names = []
    for study in studies:
        rmses = [episode.rmse() for episode in study.episodes]

        names = [study.name] * len(rmses)

        combined_rmses += rmses
        combined_names += names

    combined_rmses = np.array(combined_rmses) * 1e3  # Convert to millimeters

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title)
    sns.boxplot(
        x=combined_rmses,
        y=combined_names,
        hue=combined_names,
        palette=palette,
        saturation=saturation,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("RMSE (mm)")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="minor", left=False, right=False)

    if save_path is not None:
        assert fig is not None, "Cannot save figure when axes was passed."
        fig.savefig(save_path)
