from functools import partial
from pathlib import Path

import numpy as np
import torch.nn as nn
from rl_zoo3 import linear_schedule
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
#from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

from wandb.integration.sb3 import WandbCallback

import wandb
from src.eval import Study
from src.eval.eval_rl_v3_sim import evaluate_policy
from src.train.trainer import make_env
from src.utils import load_config, save_config

from src.environments.mpc_policy import MPCPolicy

def main():
    config = {
    # ===== Environment parameters =====
    "reward_signals": {
      "l2_norm_alignment_distance": {
        "weight": 1.0
       }
    },
    "action_mode": "delta",
    "max_quad_setting": 30.0,
    "max_quad_delta": 30.0,
    "max_steerer_delta": 6.1782e-3,
    "magnet_init_mode": np.array([10.0, -10.0, 0.0, 10.0, 0.0]),
    "incoming_mode": "random",
    "misalignment_mode": "random",
    "max_misalignment": 5e-4,
    "simulate_finite_screen": False,
    "target_beam_mode": np.zeros(4),
    "clip_magnets": True,
    
    # ===== Wrapper parameters =====
    "frame_stack": 1,
    "normalize_observation": True,
    "rescale_action": True,
    "target_threshold": None,
    "max_episode_steps": 50,
    
    # ===== RL algorithm parameters =====
    "batch_size": 64,
    "learning_rate": 0.0003,
    "lr_schedule": "constant",
    "gamma": 0.75,
    "n_envs": 1,  # Using a single environment
    "n_steps": 256,
    "ent_coef": 0.0,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "clip_range": 0.2, # Try: 0.1 to make updates more conservative.
    "clip_range_vf": None,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
    "total_timesteps": 500_000,
    
    # ===== Policy parameters =====
    "net_arch": "small",
    "activation_fn": "Tanh",
    "ortho_init": True,
    "log_std_init": 0.0,
    
    # ===== SB3 configuration =====
    "sb3_device": "auto",
    }

    # Setup wandb
    wandb.init(
        project="rl4aa25-challenge",
        sync_tensorboard=True,
        monitor_gym=True,
        config=config,
        dir=".wandb",
    )
    config = dict(wandb.config)
    config["run_name"] = wandb.run.name

    # Create a single environment (not vectorized)
    env = make_env(config)

    # Check the environment
    #check_env(env, warn=True, skip_render_check=True)

    # Setup evaluation environment
    eval_env = make_env(config, plot_episode=True, log_task_statistics=True)

    # Setup learning rate schedule if needed
    if config["lr_schedule"] == "linear":
        config["learning_rate"] = linear_schedule(config["learning_rate"])

    # Setup RL training algorithm
    model = PPO(
        #"MlpPolicy",
        MPCPolicy,  # Use the custom policy
        env,
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
            "net_arch": {
                "small": {"pi": [64, 64], "vf": [64, 64]},
                "medium": {"pi": [256, 256], "vf": [256, 256]},
            }[config["net_arch"]],
            "ortho_init": config["ortho_init"],
            "log_std_init": config["log_std_init"],
            "env": env,
        },
        device=config["sb3_device"],
        tensorboard_log=f"log/{config['run_name']}",
        verbose=1,
     )

    # Setup callbacks for evaluation and logging
    eval_callback = EvalCallback(eval_env, eval_freq=1_000, n_eval_episodes=5)
    wandb_callback = WandbCallback()

    # Train the model
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[eval_callback, wandb_callback],
    )

    # Save the model and associated configuration
    model_save_path = f"models/ea/ppo/{wandb.run.name}/model"
    config_save_path = f"models/ea/ppo/{wandb.run.name}/config"
    
    # Ensure the directory exists
    Path(f"models/ea/ppo/{wandb.run.name}").mkdir(parents=True, exist_ok=True)
    
    model.save(model_save_path)
    save_config(config, config_save_path)
    
    print(f"Model saved to {model_save_path}")
    print(f"Config saved to {config_save_path}")

    # Cleanup environments
    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()
