import gymnasium as gym
import numpy as np
import torch.nn as nn
import wandb
from gymnasium.wrappers import (
    FlattenObservation,
    FrameStack,
    RecordVideo,
    RescaleAction,
    TimeLimit,
)
from rl_zoo3 import linear_schedule
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from ..environments import cheetah_env
from ..environments.mpc_controller import MPCController
from ..utils import save_config
from ..wrappers import LogTaskStatistics, PlotEpisode, RescaleObservation


def main() -> None:
    config = {
        # Environment
        "action_mode": "delta",
        "max_quad_setting": 30.0,
        "max_quad_delta": 30.0,
        "max_steerer_delta": 6.1782e-3,
        "magnet_init_mode": np.array([10.0, -10.0, 0.0, 10.0, 0.0]),
        "incoming_mode": "random",
        "misalignment_mode": "random",
        "simulate_finite_screen": False,
        "max_misalignment": 5e-4,
        "target_beam_mode": np.zeros(4),
        "threshold_hold": 1,
        "clip_magnets": True,
        # Reward (also environment)
        "beam_param_transform": "ClippedLinear",
        "beam_param_combiner": "Mean",
        "beam_param_combiner_args": {},
        "beam_param_combiner_weights": [1, 1, 1, 1],
        "magnet_change_transform": "Sigmoid",
        "magnet_change_combiner": "Mean",
        "magnet_change_combiner_args": {},
        "magnet_change_combiner_weights": [1, 1, 1, 1, 1],
        "final_combiner": "Mean",
        "final_combiner_args": {},
        "final_combiner_weights": [3, 0.5, 0.5],
        "reward_signals": {"l2_norm_alignment_distance": {"weight": 1.0}},
        # Wrappers
        "frame_stack": 1,  # 1 means no frame stacking
        "normalize_observation": True,
        "running_obs_norm": False,
        "normalize_reward": False,  # Not really needed because normalised by design
        "rescale_action": True,
        "target_threshold": None,  # 2e-5 m is estimated screen resolution
        "max_episode_steps": 50,
        "polished_donkey_reward": False,
        # RL algorithm
        "batch_size": 64,
        "learning_rate": 0.0003,
        "lr_schedule": "constant",  # Can be "constant" or "linear"
        "gamma": 0.99,
        "n_steps": 64,
        "ent_coef": 0.0,
        "n_epochs": 10,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,  # None,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": None,
        "total_timesteps": 5_000_000,
        # Policy
        "net_arch": "small",  # Can be "small" or "medium"
        "activation_fn": "Tanh",  # Tanh, ReLU, GELU
        "ortho_init": True,  # True, False
        "log_std_init": 0.0,
        # SB3 config
        "sb3_device": "auto",
    }
    train(config)


def train(config: dict) -> None:
    # Setup wandb
    wandb.init(
        entity="msk-ipc",
        project="rl4aa-tutorial-2025-dev",
        sync_tensorboard=True,
        monitor_gym=True,
        config=config,
        dir=".wandb",
    )
    config = dict(wandb.config)
    config["run_name"] = wandb.run.name

    # Create a single environment for training (no vectorization)
    train_env = make_env(config)

    # Create a separate environment for evaluation
    eval_env = make_env(config, plot_episode=True, log_task_statistics=True)

    # Apply observation normalization if using running normalization
    if config["normalize_observation"] and config["running_obs_norm"]:
        from stable_baselines3.common.preprocessing import get_obs_shape
        from stable_baselines3.common.running_mean_std import RunningMeanStd

        class NormalizeObservation(gym.Wrapper):
            def __init__(self, env, epsilon=1e-8):
                super().__init__(env)
                self.obs_rms = RunningMeanStd(
                    shape=get_obs_shape(self.observation_space)
                )
                self.epsilon = epsilon

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                obs = self._normalize_observation(obs)
                return obs, reward, terminated, truncated, info

            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                return self._normalize_observation(obs), info

            def _normalize_observation(self, obs):
                # Update running mean and std
                self.obs_rms.update(obs)
                # Normalize observations
                return (obs - self.obs_rms.mean) / np.sqrt(
                    self.obs_rms.var + self.epsilon
                )

        train_env = NormalizeObservation(train_env)
        # Don't update statistics during evaluation
        eval_norm_env = NormalizeObservation(eval_env)
        eval_norm_env.obs_rms = train_env.obs_rms  # Share statistics
        eval_env = eval_norm_env

    # Apply reward normalization if needed
    if config["normalize_reward"]:
        from stable_baselines3.common.running_mean_std import RunningMeanStd

        class NormalizeReward(gym.Wrapper):
            def __init__(self, env, gamma=0.99, epsilon=1e-8):
                super().__init__(env)
                self.reward_rms = RunningMeanStd(shape=())
                self.gamma = gamma
                self.epsilon = epsilon
                self.return_rms = RunningMeanStd(shape=())
                self.returns = np.zeros(1)

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.returns = self.gamma * self.returns + reward
                self.reward_rms.update(self.returns)
                normalized_reward = reward / np.sqrt(self.reward_rms.var + self.epsilon)

                if terminated or truncated:
                    self.returns = np.zeros(1)

                return obs, normalized_reward, terminated, truncated, info

            def reset(self, **kwargs):
                self.returns = np.zeros(1)
                return self.env.reset(**kwargs)

        if config["normalize_reward"]:
            train_env = NormalizeReward(train_env, gamma=config["gamma"])
            # Don't update statistics during evaluation
            eval_norm_env = NormalizeReward(eval_env, gamma=config["gamma"])
            eval_norm_env.reward_rms = train_env.reward_rms  # Share statistics
            eval_env = eval_norm_env

    # Setup learning rate schedule if needed
    if config["lr_schedule"] == "linear":
        config["learning_rate"] = linear_schedule(config["learning_rate"])

    # Train using a single environment
    model = PPO(
        #"MlpPolicy",
        MPCPolicy,
        train_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        clip_range_vf=config["clip_range_vf"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        use_sde=config["use_sde"],
        sde_sample_freq=config["sde_sample_freq"],
        target_kl=config["target_kl"],
        policy_kwargs={
            "activation_fn": getattr(nn, config["activation_fn"]),
            "net_arch": {  # From rl_zoo3
                "small": {"pi": [64, 64], "vf": [64, 64]},
                "medium": {"pi": [256, 256], "vf": [256, 256]},
            }[config["net_arch"]],
            "ortho_init": config["ortho_init"],
            "log_std_init": config["log_std_init"],
        },
        device=config["sb3_device"],
        tensorboard_log=f"log/{config['run_name']}",
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=1_000,
        n_eval_episodes=5,
        best_model_save_path=f"models/ea/ppo/{wandb.run.name}/best_model",
    )
    wandb_callback = WandbCallback()

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[eval_callback, wandb_callback],
    )

    model.save(f"models/ea/ppo/{wandb.run.name}/model")

    # Save normalization parameters if using running normalization
    if config["normalize_observation"] and config["running_obs_norm"]:
        import pickle

        with open(f"models/ea/ppo/{wandb.run.name}/obs_rms.pkl", "wb") as f:
            pickle.dump(train_env.obs_rms, f)

    if config["normalize_reward"]:
        import pickle

        with open(f"models/ea/ppo/{wandb.run.name}/reward_rms.pkl", "wb") as f:
            pickle.dump(train_env.reward_rms, f)

    save_config(config, f"models/ea/ppo/{wandb.run.name}/config")

    train_env.close()
    eval_env.close()


def make_env(
    config: dict,
    record_video: bool = False,
    plot_episode: bool = False,
    log_task_statistics: bool = False,
) -> gym.Env:
    env = cheetah_env.CheetahEnv(
        generate_screen_images=plot_episode,
        incoming_mode=config["incoming_mode"],
        max_misalignment=config["max_misalignment"],
        misalignment_mode=config["misalignment_mode"],
        simulate_finite_screen=config["simulate_finite_screen"],
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        max_quad_setting=config["max_quad_setting"],
        max_quad_delta=config["max_quad_delta"],
        max_steerer_delta=config["max_steerer_delta"],
        target_beam_mode=config["target_beam_mode"],
        target_threshold=config["target_threshold"],
        # threshold_hold=config["threshold_hold"],
        clip_magnets=config["clip_magnets"],
        render_mode="rgb_array",
        reward_signals=config["reward_signals"],
    )
    # Wrap environment around a differential MPC
    env = MPCController(env)
    env = TimeLimit(env, config["max_episode_steps"])
    if plot_episode:
        env = PlotEpisode(
            env,
            save_dir=f"plots/{config['run_name']}",
            episode_trigger=lambda x: x % 5 == 0,  # Once per (5x) evaluation
            log_to_wandb=True,
        )
    if log_task_statistics:
        env = LogTaskStatistics(env)
    if config["normalize_observation"]:
        env = RescaleObservation(env, -1, 1)
    if config["rescale_action"]:
        env = RescaleAction(env, -1, 1)
    env = FlattenObservation(env)
    if config["frame_stack"] > 1:
        env = FrameStack(env, config["frame_stack"])
    env = Monitor(env)
    if record_video:
        env = RecordVideo(
            env,
            video_folder=f"recordings/{config['run_name']}",
            episode_trigger=lambda x: x % 5 == 0,  # Once per (5x) evaluation
        )
    return env


if __name__ == "__main__":
    main()
