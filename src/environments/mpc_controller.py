import gymnasium as gym
import torch
from gymnasium.spaces import Dict
from mpc.mpc import MPC, GradMethods, QuadCost


class MPCController:
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
        self.u_lower = torch.tensor(
            env.unwrapped.action_space.low,
            dtype=torch.float32
        )
        self.u_upper = torch.tensor(
            env.unwrapped.action_space.high,
            dtype=torch.float32
        )

        # Control cost matrix R (assuming diagonal as in the proposed version)
        self.R = R_scale * torch.eye(
            self.action_dim, dtype=torch.float32
        )  # Control cost (n_ctrl)

    def get_action_from_cost_params(self, state, q_diag, p):
        # Add batch dimension if missing
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, state_dim], e.g., [1, 4]
        if q_diag.dim() == 1:
            q_diag = q_diag.unsqueeze(0)  # [1, 9]
        if p.dim() == 1:
            p = p.unsqueeze(0)  # [1, 9]

        batch_size = state.shape[0]  # Should be 1 for single environment

        # Quadratic terms
        # Q is a diagonal matrix penalizing the combined state and control vector [x; u],
        # where x is the state (4D) and u is the control (5D).
        # Q as a diagonal matrix: [batch_size, 9, 9]
        Q = torch.diag_embed(q_diag)  # [1, 9, 9] for batch_size=1

        # Repeat for horizon (assuming same cost per step)
        # Running cost: [horizon, batch_size, 9, 9]
        Q_running = Q.unsqueeze(0).repeat(self.horizon, 1, 1, 1)  # [horizon, 1, 9, 9]

        # Terminal cost (using same as running cost for simplicity)
        # Terminal cost: [1, batch_size, 9, 9]
        Q_terminal = Q.unsqueeze(0)  # [1, 1, 9, 9]

        # Combine into cost tensors for QuadCost - stack them properly
        # Combine: [horizon + 1, batch_size, 9, 9]
        C = torch.cat([Q_running, Q_terminal], dim=0)  # [horizon + 1, 1, 9, 9]

        # Linear terms
        p_running = p.unsqueeze(0).repeat(self.horizon, 1, 1)  # [horizon, 1, 9]

        # Terminal cost (using same as running cost for simplicity)
        p_terminal = p.unsqueeze(0)  # [1, 1, 9]

        # Combine into cost tensors for QuadCost - stack them properly
        c = torch.cat([p_running, p_terminal], dim=0)  # [horizon + 1, 1, 9]

        # Construct cost for QuadCost, running cost for [s; u]
        # The cost is repeated over the horizon and includes a terminal cost
        cost = QuadCost(C, c)

        # Control bounds: [horizon, batch_size, action_dim]
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
            delta_u=0.1,
        )

        # Create dynamics here instead of fetching it
        dynamics = self.env.unwrapped.dynamics

        with torch.enable_grad():  # Force gradient tracking
            x_lqr, u_lqr, _ = mpc(state, cost, dynamics)

        # Detach first action for environment stepping
        action = u_lqr[0, 0].detach()

        return action
