import gymnasium as gym
import numpy as np


class RescaleObservation(gym.ObservationWrapper):
    """
    Rescales the observation of transverse tuning environments to `scaled_range`. This
    is intended as a fixed form of observation normalisation.
    """

    def __init__(
        self, env: gym.Env, min_observation: float = -1, max_observation: float = 1
    ):
        super().__init__(env)

        self.min_observation = min_observation
        self.max_observation = max_observation

        self.observation_space = gym.spaces.Dict(
            {
                key: gym.spaces.Box(
                    low=min_observation if key != "beam" else -np.inf,
                    high=max_observation if key != "beam" else np.inf,
                    shape=space.shape,
                    dtype=space.dtype,
                )
                for key, space in env.observation_space.spaces.items()
            }
        )

    def observation(self, observation: dict) -> dict:
        return {key: self._rescale(key, value) for key, value in observation.items()}

    def _rescale(self, key: str, value: np.ndarray) -> np.ndarray:
        if key == "beam":  # Exception for "beam" which has infinite range
            key = "target"  # Scale beam just like target

        return self.min_observation + (value - self.env.observation_space[key].low) * (
            self.max_observation - self.min_observation
        ) / (self.env.observation_space[key].high - self.env.observation_space[key].low)
