from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

# Import the MPC solver components
from src.environments.mpc_controller import MPCController

import torch

class MPCPolicy(ActorCriticPolicy):
    """
    Custom policy for PPO that uses NeuralCostMap with MPC for action selection.
    The policy samples cost parameters (Q diagonal and p), which are mapped to control actions via MPC.
    """

    def __init__(
        self,
        observation_space: spaces.Space,  # 13D
        action_space: spaces.Box,  # 18D
        lr_schedule: Callable[[float], float],
        env=None,
        state_dim=4,  # TODO: IS IT NEEDED HERE??
        action_dim=5, # TODO: IS IT NEEDED HERE??
        horizon=15,  # typical range: 10–20 to consider longer-term effects
        lqr_iter=25, # typical range: 5–50 iterations
        R_scale=0.5, # 0.1–1.0 to penalize control effort more, encouraging smoother actions
        target_state_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs
    ):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        # Define defaults
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        # Environment
        self.env = env

        # Redefine action_space as cost parameters (18D for Q diagonal + p)
        self.state_dim = state_dim  # Use passed state_dim
        self.action_dim = action_dim  # Use passed action_dim
        self.combined_dim = self.state_dim + self.action_dim

        # MPC setup
        self.horizon = horizon
        self.lqr_iter = lqr_iter
        self.R_scale = R_scale

        # Function to extract target state from observation
        self.target_state_fn = (
            target_state_fn
            if target_state_fn is not None
            else lambda x: x[:, -self.state_dim:]
        )

    def _build_mlp_extractor(self) -> None:
        """Build the MLP extractor for actor and critic."""
        super()._build_mlp_extractor()
        assert hasattr(self, "mlp_extractor"), "MLP extractor not initialized"
        assert self.mlp_extractor is not None, "MLP extractor is None"

    def _get_action_dist_from_latent(
        self, latent_pi: torch.Tensor
    ) -> DiagGaussianDistribution:
        """Create a diagonal Gaussian distribution from the actor’s latent features."""
        mean_actions = self.action_net(latent_pi)  # [batch_size, 18]
        dist = DiagGaussianDistribution(action_dim=self.combined_dim * 2)  # 18
        dist.proba_distribution(
            mean_actions=mean_actions,
            log_std=self.log_std,  # Inherited from ActorCriticPolicy
        )

        return dist

    def forward(self, obs: torch.Tensor, deterministic=False):
        """
        Forward pass: Predict cost parameters and map to control actions via MPC.

        Sample and return 18D cost parameters.
        The wrapper will convert these to 5D control actions.
        """
        # Extract features
        latent_pi, latent_vf = self.mlp_extractor(obs)  # Use mlp_extractor directly

        # Get cost parameters distribution from actor
        dist = self._get_action_dist_from_latent(latent_pi)

        # The differential-MPC cost parameters
        cost_params = dist.get_actions(deterministic=deterministic)  # [batch_size, 18]

        # Log probability of the cost parameters (not actions)
        log_prob = dist.log_prob(cost_params) if not deterministic else None

        # Get value estimate from critic
        values = self.value_net(latent_vf)

        # The cost_params represent the non-control actions
        return cost_params, values, log_prob # output the cost_params

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate the log probability of actions (cost parameters) and compute values.
        Evaluate log probabilities of the 18D cost parameters.
        """
        latent_pi, latent_vf = self.mlp_extractor(obs)

        # Get predicted cost parameters from actor
        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)  # actions are 18D cost parameters

        # Value estimate
        values = self.value_net(latent_vf)

        return (
            values,
            log_prob,
            dist.entropy(),
        )
