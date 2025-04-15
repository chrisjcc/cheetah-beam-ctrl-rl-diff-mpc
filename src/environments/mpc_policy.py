from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

# Import the MPC solver components
from src.environments.mpc_controller import MPCController
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class MPCPolicy(ActorCriticPolicy):
    """
    Custom policy for PPO that uses NeuralCostMap with MPC for action selection.
    The policy samples cost parameters (Q diagonal and p), which are mapped to control actions via MPC.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Callable[[float], float],
        env=None,
        state_dim=4,
        action_dim=5,
        horizon=5,
        lqr_iter=500,
        R_scale=0.01,
        target_state_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs
    ):
        # Define defaults
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        # Redefine action_space as cost parameters (18D for Q diagonal + p)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.combined_dim = state_dim + action_dim
        action_space = spaces.Box(
            low=-10, high=10, shape=(self.combined_dim * 2,), dtype=np.float32
        )  # 18D

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        # Environment and MPC setup
        self.env = env
        self.horizon = horizon
        self.lqr_iter = lqr_iter
        self.R_scale = R_scale

        self.mpc_controller = MPCController(self.env, horizon, lqr_iter, R_scale)

        # Function to extract target state from observation
        self.target_state_fn = (
            target_state_fn
            if target_state_fn is not None
            else lambda x: x[:, state_dim: state_dim * 2]
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
        """
        # Extract features
        latent_pi, latent_vf = self.mlp_extractor(obs)  # Use mlp_extractor directly

        # Extract current and target states
        current_state = obs[:, : self.state_dim].to(torch.float32)
        target_state = self.target_state_fn(obs).to(torch.float32)
        current_state.requires_grad_(True)
        target_state.requires_grad_(True)

        # Get cost parameters distribution from actor
        dist = self._get_action_dist_from_latent(latent_pi)
        cost_params = dist.get_actions(deterministic=deterministic)  # [batch_size, 18]
        q_diag = torch.exp(
            cost_params[:, : self.combined_dim]
        )  # Ensure positive diagonal
        p = cost_params[:, self.combined_dim:]

        # In MPCPolicy.forward
        action = self.mpc_controller.get_action_from_cost_params(
            current_state, target_state, q_diag, p
        )

        # Get value estimate from critic
        values = self.value_net(latent_vf)

        # Log probability of the cost parameters (not actions)
        log_prob = dist.log_prob(cost_params) if not deterministic else None

        return action, values, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate the log probability of actions (cost parameters) and compute values.
        """
        latent_pi, latent_vf = self.mlp_extractor(obs)

        # Extract current and target states
        current_state = obs[:, : self.state_dim].to(torch.float32)
        target_state = self.target_state_fn(obs).to(torch.float32)

        # Get predicted cost parameters from actor
        dist = self._get_action_dist_from_latent(latent_pi)
        predicted_cost_params = dist.get_actions(deterministic=True)  # [batch_size, 18]

        # Since `actions` from the rollout buffer are control actions (5D),
        # we need to compare them with the MPC output, not the cost parameters
        # directly
        q_diag = torch.exp(predicted_cost_params[:, : self.combined_dim])
        p = predicted_cost_params[:, self.combined_dim:]

        predicted_actions = self.mpc_controller.get_action_from_cost_params(
            current_state, target_state, q_diag, p
        )
        predicted_actions = torch.tensor(
            predicted_actions, dtype=torch.float32, device=actions.device
        )

        # Simplified log probability: Gaussian approximation around predicted control actions
        action_var = torch.ones_like(actions) * 0.1  # Fixed variance
        log_prob = -0.5 * torch.sum(
            ((actions - predicted_actions) ** 2) / action_var, dim=-1
        )

        # Value estimate
        values = self.value_net(latent_vf)

        return (
            values,
            log_prob,
            None,
        )  # Entropy is None as it’s not directly meaningful here
