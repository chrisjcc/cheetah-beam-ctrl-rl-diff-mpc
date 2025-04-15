from typing import Optional
import numpy as np
import math


class Rewarder:
    def __init__(self, config):
        """
        Initialize the Rewarder with a configuration dictionary.

        :param config: Dictionary with reward labels as keys and weights as values
        """
        self.config = config
        self.info = {}
        self.weights = np.array([1, 1, 2, 2])  # Configurable weights for objective calculation

    def l1_norm_variance(self, state: np.ndarray, target: np.ndarray):
        """
        Calculate L1 norm (Manhattan distance) variance.

        :param state: Current state
        :param target: Target state
        :return: L1 norm variance reward
        """
        difference = state - target

        return -float(np.linalg.norm(difference, ord=1))

    def l2_norm_variance(self, state: np.ndarray, target: np.ndarray):
        """
        Calculate L2 norm (Euclidean distance) variance.

        :param state: Current state
        :param target: Target state
        :return: L2 norm variance reward
        """
        difference = state - target

        return -float(np.linalg.norm(difference))

    def l2_norm_alignment_distance(self, state: np.ndarray, target: np.ndarray):
        """
        Calculate Euclidean distance to target.

        :param state: Current state
        :param target: Target state
        :return: L2-norm target distance reward
        """
        distance = float(np.linalg.norm(state - target))

        return -distance / self.max_distance

    def inverse_alignment_distance(self, state: np.ndarray, target: np.ndarray):
        """
        Calculate inverse distance reward.

        :param state: Current state
        :param target: Target state
        :return: Inverse distance reward
        """
        distance = float(np.linalg.norm(state - target))

        return 1.0 / (1.0 + distance / self.max_distance)

    def quadratic_inverse_alignment_distance(self, state: np.ndarray, target: np.ndarray):
        """
        Calculate quadratic inverse distance reward.

        :param state: Current state
        :param target: Target state
        :return: Quadratic inverse distance reward
        """
        distance = float(np.linalg.norm(state - target))

        return 1.0 / (1.0 + distance / self.max_distance)**2

    def exponential_l2_norm_alignment_distance(self, state: np.ndarray, target: np.ndarray):
        """
        Calculate exponential distance reward.

        :param state: Current state
        :param target: Target state
        :return: Exponential distance reward
        """
        distance = float(np.linalg.norm(state - target))

        return np.exp(-distance / self.max_distance)

    def logarithmic_scaling_inverse_alignment_distance(self, state: np.ndarray, target: np.ndarray):
        """
        Calculate logarithmic inverse distance reward.

        :param state: Current state
        :param target: Target state
        :return: Logarithmic inverse distance reward
        """
        distance = float(np.linalg.norm(state - target))

        return 1.0 / np.log(1.0 + distance / self.max_distance)

    def logarithmic_scaling_alignment_distance(self, state: np.ndarray, target: np.ndarray):
        """
        Calculate logarithmic distance reward.

        :param state: Current state
        :param target: Target state
        :return: Logarithmic inverse distance reward
        """
        distance = float(np.linalg.norm(state - target))

        return -np.log(distance / self.max_distance)

    def progress_alignment_distance(self, state: np.ndarray, prev_state: np.ndarray, target: np.ndarray):
        """
        Calculate progress towards target.

        :param state: Current state
        :param prev_state: Previous state
        :param target: Target state
        :return: Progress reward
        """
        prev_dist = float(np.linalg.norm(prev_state - target)) / self.max_distance
        curr_dist = float(np.linalg.norm(state - target)) / self.max_distance

        return prev_dist - curr_dist

    def objective_progress(self, state: np.ndarray, previous: np.ndarray, target: np.ndarray):
        """
        Calculate the reward based on the change in the objective value.

        Sources:
          - Eq. (5)-(6). https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.27.054601
          - Eq. (4), https://arxiv.org/pdf/2306.03739
          - No eq. numbers, https://proceedings.mlr.press/v162/kaiser22a/kaiser22a.pdf
        """
        # Compute the current objective value based on the current actuator state
        objective = self._objective_fn(state, target)

        # Compute the previous objective value based on the previous actuator state
        previous_objective = self._objective_fn(previous, target)

        # Reward is the difference between the previous and current objective values
        reward = previous_objective - objective

        # Return the reward, ensuring it's at least double the negative value if it’s less than zero
        return reward if reward > 0 else 2 * reward

    def _objective_fn(self, state: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate the objective function value based on the difference between
        achieved and desired states.
        """
        offset = state - target

        # The original log formulation with weighted absolute error
        return np.log((self.weights * np.abs(offset)).sum())

    def compute_reward(self,
            state: np.ndarray,
            target: np.ndarray, 
            height: float, 
            width: float,
            prev_state: Optional[np.ndarray] = None, 
        ):
        """
        Compute total reward based on configured reward methods.

        :param state: Current state
        :param prev_state: Previous state
        :param target: Target state
        :return: Total weighted reward
        """
        # Reset info dictionary for the current step
        self.info = {}
        total_reward = 0
        reward_components = {}

        # maximum stance from the center of the half width × height rectangle to a corner
        self.max_distance = float(np.linalg.norm([height, width]))

        for reward_label, weight in self.config.items():
            if reward_label == "l1_norm_variance":
                reward = self.l1_norm_variance(state, target)
            elif reward_label == "l2_norm_variance":
                reward = self.l2_norm_variance(state, target)
            elif reward_label == "l2_norm_alignment_distance":
                 reward = self.l2_norm_alignment_distance(state, target)
            elif reward_label == "normalized_alignment_distance":
                 reward = self.normalized_alignment_distance(state, target)
            elif reward_label == "progress_alignment_distance":
                reward = self.progress_alignment_distance(state, prev_state, target)
            elif reward_label == "inverse_alignment_distance":
                reward = self.inverse_alignment_distance(state, target)
            elif reward_label == "quadratic_inverse_alignment_distance":
                reward = self.quadratic_inverse_alignment_distance(state, target)
            elif reward_label == "exponential_l2_norm_alignment_distance":
                reward = self.exponential_l2_norm_alignment_distance(state, target)
            elif reward_label == "logarithmic_scaling_inverse_alignment_distance":
                reward = self.logarithmic_scaling_inverse_alignment_distance(state, target)
            elif reward_label == "logarithmic_scaling_alignment_distance":
                reward = self.logarithmic_scaling_alignment_distance(state, target)
            elif reward_label == "objective_progress":
                reward = self.objective_progress(state, prev_state, target)
            else:
                raise ValueError(f"Unknown reward method: {reward_label}")

            # Store individual reward components
            reward_components[reward_label] = float(reward)
            total_reward += weight * reward

        # Update info dictionary with reward details
        self.info.update({
            "reward_components": reward_components,
            "reward_weights": list(self.config.values()),
        })

        return float(total_reward)

    def get_info(self):
        """
        Retrieve the current step's info dictionary.

        :return: Dictionary of reward information
        """
        return self.info
