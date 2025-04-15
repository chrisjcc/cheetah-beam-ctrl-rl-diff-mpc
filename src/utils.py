import os
from datetime import datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import wandb
import yaml
from stable_baselines3.common.callbacks import BaseCallback

plt.style.use(["science", "nature", "no-latex"])


def evaluate_mae(observations) -> tuple[plt.Figure, plt.Axes]:
    maes = np.array(
        [np.mean(np.abs(obs["beam"] - obs["target"])) for obs in observations]
    )

    fig, ax = plt.subplots()
    ax.semilogy(maes * 1000)
    ax.set_ylabel("Mean Absolute Error (mm)")
    ax.set_xlabel("Step")

    print(f"Minimum MAE:                {min(maes) * 1000:.3f}  mm")
    print(f"Sum of MAE over all steps:  {np.sum(maes) * 1000:.3f} mm")

    return fig, ax


def load_config(path: str) -> dict:
    """
    Load a training setup config file to a config dictionary. The config file must be a
    `.yaml` file. The `path` argument to this function should be given without the file
    extension.
    """
    with open(f"{path}.yaml", "r") as f:
        data = yaml.load(f.read(), Loader=yaml.Loader)
    return data


def plot_beam_history(ax, observations, before_reset=None):
    mu_x = np.array([obs["beam"][0] for obs in observations])
    sigma_x = np.array([obs["beam"][1] for obs in observations])
    mu_y = np.array([obs["beam"][2] for obs in observations])
    sigma_y = np.array([obs["beam"][3] for obs in observations])

    if before_reset is not None:
        mu_x = np.insert(mu_x, 0, before_reset[0])
        sigma_x = np.insert(sigma_x, 0, before_reset[1])
        mu_y = np.insert(mu_y, 0, before_reset[2])
        sigma_y = np.insert(sigma_y, 0, before_reset[3])

    target_beam = observations[0]["target"]

    start = 0 if before_reset is None else -1
    steps = np.arange(start, len(observations))

    ax.set_title("Beam Parameters")
    ax.set_xlim([start, len(observations) + 1])
    ax.set_xlabel("Step")
    ax.set_ylabel("(mm)")
    ax.plot(steps, mu_x * 1e3, label=r"$\mu_x$", c="tab:blue")
    ax.plot(steps, [target_beam[0] * 1e3] * len(steps), ls="--", c="tab:blue")
    ax.plot(steps, sigma_x * 1e3, label=r"$\sigma_x$", c="tab:orange")
    ax.plot(steps, [target_beam[1] * 1e3] * len(steps), ls="--", c="tab:orange")
    ax.plot(steps, mu_y * 1e3, label=r"$\mu_y$", c="tab:green")
    ax.plot(steps, [target_beam[2] * 1e3] * len(steps), ls="--", c="tab:green")
    ax.plot(steps, sigma_y * 1e3, label=r"$\sigma_y$", c="tab:red")
    ax.plot(steps, [target_beam[3] * 1e3] * len(steps), ls="--", c="tab:red")
    ax.legend()
    ax.grid(True)


def plot_screen_image(ax, img, screen_resolution, pixel_size, title="Beam Image"):
    screen_size = screen_resolution * pixel_size

    ax.set_title(title)
    ax.set_xlabel("(mm)")
    ax.set_ylabel("(mm)")
    ax.imshow(
        img,
        vmin=0,
        aspect="equal",
        interpolation="none",
        extent=(
            -screen_size[0] / 2 * 1e3,
            screen_size[0] / 2 * 1e3,
            -screen_size[1] / 2 * 1e3,
            screen_size[1] / 2 * 1e3,
        ),
    )


def plot_quadrupole_history(ax, observations, before_reset=None):
    areamqzm1 = [obs["magnets"][0] for obs in observations]
    areamqzm2 = [obs["magnets"][1] for obs in observations]
    areamqzm3 = [obs["magnets"][3] for obs in observations]

    if before_reset is not None:
        areamqzm1 = [before_reset[0]] + areamqzm1
        areamqzm2 = [before_reset[1]] + areamqzm2
        areamqzm3 = [before_reset[3]] + areamqzm3

    start = 0 if before_reset is None else -1
    steps = np.arange(start, len(observations))

    ax.set_title("Quadrupoles")
    ax.set_xlim([start, len(observations) + 1])
    ax.set_xlabel("Step")
    ax.set_ylabel("Strength (1/m^2)")
    ax.plot(steps, areamqzm1, label="AREAMQZM1")
    ax.plot(steps, areamqzm2, label="AREAMQZM2")
    ax.plot(steps, areamqzm3, label="AREAMQZM3")
    ax.legend()
    ax.grid(True)


def plot_steerer_history(ax, observations, before_reset=None):
    areamcvm1 = np.array([obs["magnets"][2] for obs in observations])
    areamchm2 = np.array([obs["magnets"][4] for obs in observations])

    if before_reset is not None:
        areamcvm1 = np.insert(areamcvm1, 0, before_reset[2])
        areamchm2 = np.insert(areamchm2, 0, before_reset[4])

    start = 0 if before_reset is None else -1
    steps = np.arange(start, len(observations))

    ax.set_title("Steerers")
    ax.set_xlabel("Step")
    ax.set_ylabel("Kick (mrad)")
    ax.set_xlim([start, len(observations) + 1])
    ax.plot(steps, areamcvm1 * 1e3, label="AREAMCVM1")
    ax.plot(steps, areamchm2 * 1e3, label="AREAMCHM2")
    ax.legend()
    ax.grid(True)


def remove_if_exists(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def save_config(data: dict, path: str) -> None:
    """
    Save a training setup config to a `.yaml` file. The `path` argument to this function
    should be given without the file extension.
    """
    with open(f"{path}.yaml", "w") as f:
        yaml.dump(data, f)


class CheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq,
        save_path,
        name_prefix="rl_model",
        save_env=False,
        env_name_prefix="vec_normalize",
        save_replay_buffer=False,
        replay_buffer_name_prefix="replay_buffer",
        delete_old_replay_buffers=True,
        verbose=0,
    ):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_env = save_env
        self.env_name_prefix = env_name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.replay_buffer_name_prefix = replay_buffer_name_prefix
        self.delete_old_replay_buffers = delete_old_replay_buffers

    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save model
            path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps"
            )
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")

            # Save env (VecNormalize wrapper)
            if self.save_env:
                path = os.path.join(
                    self.save_path,
                    f"{self.env_name_prefix}_{self.num_timesteps}_steps.pkl",
                )
                self.training_env.save(path)
                if self.verbose > 1:
                    print(f"Saving environment to {path[:-4]}")

            # Save replay buffer
            if self.save_replay_buffer:
                path = os.path.join(
                    self.save_path,
                    f"{self.replay_buffer_name_prefix}_{self.num_timesteps}_steps",
                )
                self.model.save_replay_buffer(path)
                if self.verbose > 1:
                    print(f"Saving replay buffer to {path}")

                if self.delete_old_replay_buffers and hasattr(self, "last_saved_path"):
                    remove_if_exists(self.last_saved_path + ".pkl")
                    if self.verbose > 1:
                        print(f"Removing old replay buffer at {self.last_saved_path}")

                self.last_saved_path = path

        return True


class SLURMRescheduleCallback(BaseCallback):
    def __init__(self, reserved_time, safety=timedelta(minutes=1), verbose=0):
        super().__init__(verbose)
        self.allowed_time = reserved_time - safety
        self.t_start = datetime.now()
        self.t_last = self.t_start

    def _on_step(self):
        t_now = datetime.now()
        passed_time = t_now - self.t_start
        dt = t_now - self.t_last
        self.t_last = t_now
        if passed_time + dt > self.allowed_time:
            os.system(
                "sbatch"
                f" --export=ALL,WANDB_RESUME=allow,WANDB_RUN_ID={wandb.run.id} td3.sh"
            )
            if self.verbose > 1:
                print("Scheduling new batch job to continue training")
            return False
        else:
            if self.verbose > 1:
                print(
                    f"Continue running with this SLURM job (passed={passed_time} /"
                    f" allowed={self.allowed_time} / dt={dt})"
                )
            return True


def deep_equal(obj1: Any, obj2: Any) -> bool:
    """Recursively checks if two objects are equal."""
    if type(obj1) is not type(obj2):
        return False

    if isinstance(obj1, dict):
        if obj1.keys() != obj2.keys():
            return False
        return all(deep_equal(obj1[key], obj2[key]) for key in obj1)

    if isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        return all(deep_equal(item1, item2) for item1, item2 in zip(obj1, obj2))

    if isinstance(obj1, np.ndarray):
        return np.array_equal(obj1, obj2)

    return obj1 == obj2
