from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation, FrameStack, RescaleAction, TimeLimit
from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm

from src.environments import cheetah_env

from src.trial import Trial, load_trials
from src.wrappers import RecordEpisode, RescaleObservation


def evaluate_on_trial(
    trial_index: int,
    trial: Trial,
    model: BaseAlgorithm,
    config: dict,
    write_data: bool = True,
) -> None:
    """
    Evaluate a policy on a single trial.

    :param trial_index: The index of the trial.
    :param trial: The trial to evaluate on.
    :param model: The policy model to evaluate.
    :param config: The configuration dictionary used when training the policy.
    :param write_data: Whether to write the episode data to disk.
    """

    policy_name = config["run_name"]

    # Create the environment
    env = cheetah_env.CheetahEnv(
        incoming_mode=trial.incoming_beam,
        max_misalignment=5e-4,
        misalignment_mode=trial.misalignments,
        simulate_finite_screen=False,
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        max_quad_setting=config["max_quad_setting"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
        target_beam_mode=np.zeros(4),
        target_threshold=None,
        threshold_hold=5,
        clip_magnets=True,
        reward_signals=config["reward_signals"]
    )
    env = TimeLimit(env, 150)
    if write_data:
        env = RecordEpisode(
            env,
            save_dir=(f"data/{policy_name}/problem_{trial_index:03d}"),
        )
    if config["normalize_observation"]:
        env = RescaleObservation(env, -1, 1)
    if config["rescale_action"]:
        env = RescaleAction(env, -1, 1)
    env = FlattenObservation(env)
    if config["frame_stack"] > 1:
        env = FrameStack(env, config["frame_stack"])

    # Actual optimisation
    observation, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()


def generate_trials(num: int, seed: int | None = None) -> list[Trial]:
    """
    Generate a set of random trials for evaluation. Each trial consists of a
    random incoming beam, target beam, quadrupole misalignment and screen
    misalignment.

    :param num: The number of trials to generate.
    :param seed: The seed to use for generating the trials.
    :return: A list of trials.
    """

    assert num > 0

    target_beam_space = spaces.Box(
        low=np.array([-2e-3, 0, -2e-3, 0], dtype=np.float32),
        high=np.array([2e-3, 2e-3, 2e-3, 2e-3], dtype=np.float32),
        seed=seed,
    )
    incoming_beam_space = spaces.Box(
        low=np.array(
            [
                80e6,
                -1e-3,
                -1e-4,
                -1e-3,
                -1e-4,
                1e-5,
                1e-6,
                1e-5,
                1e-6,
                1e-6,
                1e-4,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [160e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3],
            dtype=np.float32,
        ),
        seed=seed,
    )
    misalignment_space = spaces.Box(
        low=-5e-4, high=5e-4, shape=(8,), dtype=np.float32, seed=seed
    )
    initial_magnet_space = spaces.Box(
        low=np.array([-30, -30, -6.1782e-3, -30, -6.1782e-3]),
        high=np.array([30, 30, 6.1782e-3, 30, 6.1782e-3], dtype=np.float32),
        seed=seed,
    )

    trials = [
        Trial(
            target_beam=target_beam_space.sample(),
            incoming_beam=incoming_beam_space.sample(),
            misalignments=misalignment_space.sample(),
            initial_magnets=initial_magnet_space.sample(),
        )
        for _ in range(num)
    ]

    return trials


def evaluate_policy(
    model: BaseAlgorithm, config: dict, write_data: bool = True, seed: int | None = None
) -> None:
    """
    Evaluate a policy on a set of trials (different incoming beams, quadrupole
    misalignments and screen misalignments).

    :param model: The policy model to evaluate.
    :param config: The configuration dictionary used when training the policy.
    :param write_data: Whether to write the episode data to disk.
    :param seed: The seed to use for generating the trials.
    """

    trials = generate_trials(num=20, seed=seed)

    for i, trial in enumerate(trials):
        evaluate_on_trial(i, trial, model, config, write_data)


def main():
    trials = load_trials(Path("data/trials.yaml"))

    with ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(evaluate_on_trial, range(len(trials)), trials), total=300)


if __name__ == "__main__":
    main()
