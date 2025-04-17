import numpy as np
import torch
from gymnasium import Wrapper, spaces


class MPCWrapper(Wrapper):
    def __init__(self, env, mpc_controller):
        super().__init__(env)
        self.mpc_controller = mpc_controller

        # Redefine action_space to match policy's output: 18D cost parameters
        self.action_space = spaces.Box(low=-10, high=10, shape=(18,), dtype=np.float32)

        # Store the last observation to access the current state
        self.last_obs = None

        self.state_dim = 4  # TODO: Generalize!!!

    def step(self, action):
        """
        Convert 18D cost parameters to 5D control actions and step the environment.
        Action: numpy array of shape (18,) from PPO policy.
        """
        if self.last_obs is None:
            raise ValueError("Environment must be reset before stepping.")

        # Extract current state from the last observation
        # Assuming observation is a dict with "beam" as the state
        obs = torch.tensor(self.last_obs, dtype=torch.float32)
        current_state = obs[: self.state_dim].to(torch.float32)

        # Convert numpy action to torch tensor
        action = torch.tensor(action, dtype=torch.float32)
        q_diag = torch.exp(action[:9])  # First 9 elements for q_diag
        p = action[9:]  # Last 9 elements for p

        # Compute 5D control action using MPC
        control_action = self.mpc_controller.get_action_from_cost_params(
            current_state, q_diag, p
        )
        control_action = (
            control_action.detach().cpu().numpy()
        )  # Convert to numpy for env

        # Step the underlying environment with the 5D control action
        next_obs, reward, terminated, truncated, info = self.env.step(control_action)

        self.last_obs = next_obs  # Update last observation

        return next_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and store the initial observation."""
        obs, info = self.env.reset(**kwargs)

        self.last_obs = obs  # Store the dictionary for internal use
        return obs, info
