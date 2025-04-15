import gymnasium as gym
import numpy as np
import wandb
from gymnasium import logger

from ..eval import Episode


class LogTaskStatistics(gym.Wrapper):
    """
    Log the results of the transverse beam tuning to WandB.

    The metrics logged are:
     - Final MAE
     - RMSE over episode
     - Number of steps to convergence
     - Number of steps to threshold
    """

    def __init__(self, env, prefix: str = "task"):
        super().__init__(env)

        self.prefix = prefix

        self.episode_id = 0
        self.is_recording = False

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)

        self.is_recording = True

        self.observations = [observation]
        self.rewards = []
        self.terminateds = []
        self.truncateds = []
        self.infos = []
        self.actions = []

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if self.is_recording:
            self.observations.append(observation)
            self.rewards.append(reward)
            self.terminateds.append(terminated)
            self.truncateds.append(truncated)
            self.infos.append(info)
            self.actions.append(action)

            if terminated or truncated:
                self._log_episode()
                self.is_recording = False

        if terminated or truncated:
            self.episode_id += 1

        return observation, reward, terminated, truncated, info

    def close(self):
        super().close()

        if self.is_recording:
            self._log_episode()

    def _log_episode(self):
        """Logs episode statistics to WandB."""
        if len(self.observations) < 2:  # No data to plot
            logger.warn(
                f"Unable to save episode plot for {self.episode_id = } because the"
                " episode was too short."
            )
            return

        episode = Episode(
            observations=self.observations,
            rewards=self.rewards,
            terminateds=self.terminateds,
            truncateds=self.truncateds,
            infos=self.infos,
            actions=self.actions,
        )
        (
            beam_reward,
            on_screen_reward,
            magnet_change_reward,
        ) = self._compute_sum_of_reward_components(episode)

        final_beam_parameter_errors = np.abs(
            episode.observations[-1]["beam"] - episode.observations[-1]["target"]
        )

        wandb.log(
            {
                f"{self.prefix}/final_mae": episode.final_mae(),
                f"{self.prefix}/rmse": episode.rmse(),
                f"{self.prefix}/steps_to_convergence": episode.steps_to_convergence(
                    threshold=4e-5, use_min_mae=False
                ),
                f"{self.prefix}/steps_to_threshold": episode.steps_to_threshold(
                    threshold=4e-5, use_min_mae=False
                ),
                f"{self.prefix}/beam_reward": beam_reward,
                f"{self.prefix}/on_screen_reward": on_screen_reward,
                f"{self.prefix}/magnet_change_reward": magnet_change_reward,
                f"{self.prefix}/final_mu_x_error": final_beam_parameter_errors[0],
                f"{self.prefix}/final_sigma_x_error": final_beam_parameter_errors[1],
                f"{self.prefix}/final_mu_y_error": final_beam_parameter_errors[2],
                f"{self.prefix}/final_sigma_y_error": final_beam_parameter_errors[3],
            }
        )

    def _compute_sum_of_reward_components(
        self, episode: Episode
    ) -> tuple[float, float, float]:
        """
        Compute the sum of the reward components over the episode. Returns (beam reward,
        on screen reward, magnet change reward).
        """
        beam_reward = 0
        on_screen_reward = 0
        magnet_change_reward = 0
        for info in episode.infos[1:]:
            beam_reward += (
                info["beam_reward"] if info.get("beam_reward") is not None else 0
            )
            on_screen_reward += (
                info["on_screen_reward"]
                if info.get("on_screen_reward") is not None
                else 0
            )
            magnet_change_reward += (
                info["magnet_change_reward"]
                if info.get("magnet_change_reward") is not None
                else 0
            )

        return beam_reward, on_screen_reward, magnet_change_reward
