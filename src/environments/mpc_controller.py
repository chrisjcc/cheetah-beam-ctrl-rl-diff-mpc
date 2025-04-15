import gymnasium as gym
import torch
from gymnasium.spaces import Dict
from mpc.mpc import MPC, GradMethods, QuadCost


class MPCController(gym.Wrapper):
    """
    Model Predictive Control (MPC) based controller for beam control.
    This class encapsulates the MPC algorithm logic, separating it from the environment.

    MPC controller returns both the immediate action and the full predicted trajectory
    (states and actions over the horizon) to leverage backpropagation through
    the entire MPC computation.
    """

    def __init__(self, env, horizon=5, lqr_iter=5, R_scale=0.01):
        """
        Initialize the MPC controller.

        Args:
            dynamics: The beam dynamics model (BeamDynamics)
            action_space: The environment's action space (defines control bounds)
            horizon: The prediction horizon for MPC
            lqr_iter: Number of LQR iterations
        """
        # MPC parameters
        self.env = env

        if isinstance(self.env.unwrapped.observation_space, Dict):
            self.state_dim = self.env.unwrapped.observation_space.spaces["beam"].shape[
                0
            ]  # 4, [mu_x, sigma_x, mu_y, sigma_y]
        else:
            self.state_dim = self.env.unwrapped.observation_space.shape[
                0
            ]  # 4, [mu_x, sigma_x, mu_y, sigma_y]

        self.action_dim = self.env.unwrapped.action_space.shape[
            0
        ]  # 5, [k1_Q1, k1_Q2, angle_CV, k1_Q3, angle_CH]
        self.horizon = horizon
        self.lqr_iter = lqr_iter
        self.R_scale = R_scale

        # Action bounds as tensors
        self.u_lower = torch.tensor(env.unwrapped.action_space.low, dtype=torch.float32)
        self.u_upper = torch.tensor(
            env.unwrapped.action_space.high, dtype=torch.float32
        )

        # Control cost matrix R (assuming diagonal as in the proposed version)
        self.R = R_scale * torch.eye(
            self.action_dim, dtype=torch.float32
        )  # Control cost (n_ctrl)

    def reset(self, **kwargs):
        """
        Explicitly initialize the internal BeamDynamics object and segment parameters.
        Should be called after env.reset() from an external handler.
        """
        obs, info = self.env.unwrapped.reset(**kwargs)

        return obs, info

    def get_action_from_cost_params(self, state, target, q_diag, p):
        batch_size = state.shape[0]

        Q = torch.diag_embed(q_diag)  # [batch_size, 9, 9]

        # Repeat for horizon (assuming same cost per step)
        Q_running = Q.unsqueeze(0).repeat(self.horizon, 1, 1, 1)
        p_running = p.unsqueeze(0).repeat(self.horizon, 1, 1)

        # Terminal cost (using same as running cost for simplicity)
        Q_terminal = Q.unsqueeze(0)
        p_terminal = p.unsqueeze(0)

        # Combine into cost tensors for QuadCost - stack them properly
        C = torch.cat([Q_running, Q_terminal], dim=0)
        c = torch.cat([p_running, p_terminal], dim=0)

        # Construct cost for QuadCost, running cost for [s; u]
        cost = QuadCost(C, c)

        # Control bounds
        u_lower = (
            self.u_lower.unsqueeze(0)
            .unsqueeze(0)
            .repeat(self.horizon, batch_size, 1)
            .to(state.device)
        )
        u_upper = (
            self.u_upper.unsqueeze(0)
            .unsqueeze(0)
            .repeat(self.horizon, batch_size, 1)
            .to(state.device)
        )

        # Initialize and solve MPC
        mpc = MPC(
            n_state=self.state_dim,
            n_ctrl=self.action_dim,
            T=self.horizon,
            u_lower=u_lower,
            u_upper=u_upper,
            lqr_iter=self.lqr_iter,
            grad_method=GradMethods.AUTO_DIFF,
            verbose=1,  # Debug MPC internals
            backprop=True,
        )

        # Create dynamics here instead of fetching it
        dynamics = self.env.unwrapped.dynamics

        with torch.enable_grad():  # Force gradient tracking
            x_lqr, u_lqr, _ = mpc(state, cost, dynamics)

        # Detach first action for environment stepping
        action = u_lqr[0, 0].detach()

        return action
