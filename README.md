# cheetah-beam-ctrl-rl-diff-mpc
Reinforcement learning-based beam control using differentiable MPC and Cheetah accelerator simulation.

## Acknowledged and Cited Work

This project builds on the following frameworks and methods. If you use this repository, please ensure you cite the original authors as specified below.

### 🧠 Actor-Critic MPC with Differentiable Optimization


We implement a reinforcement learning (RL) solution using the Proximal Policy Optimization (PPO) algorithm from Stable-Baselines3 to control beam parameters in a particle accelerator, specifically within the ARES Experimental Area (AREA) lattice. The system leverages a differentiable simulator (Cheetah) to model beam dynamics, and you’ve integrated Model Predictive Control (MPC) to map high-dimensional cost parameters (learned by the RL policy) to low-dimensional control actions (magnet settings). The key components are:

- **CheetahEnv**: A Gym-compatible environment that simulates the accelerator lattice using DifferentialAREASegment and BeamDynamics. It defines the observation space (beam parameters, magnet settings, target), action space (magnet settings or their deltas), and reward function (based on beam alignment and focus).
- **MPCController**: Implements an MPC solver using the mpc.pytorch library to compute optimal control actions (5D magnet settings) based on cost parameters (18D, comprising q_diag and p) provided by the RL policy.
- **MPCPolicy**: A custom ActorCriticPolicy for PPO that outputs 18D cost parameters (q_diag and p) instead of direct control actions. These parameters define a quadratic cost function used by the MPC solver.
- **MPCWrapper**: A Gym wrapper that interfaces between the RL policy (outputting 18D cost parameters) and the environment (expecting 5D control actions) by using the MPCController to convert cost parameters to control actions.
- **BeamDynamics**: A differentiable model of beam dynamics that maps the current state and control inputs to the next state, used by the MPC solver for trajectory optimization.
- **Trainer/Runner**: Scripts to train the PPO agent, configure the environment, and set hyperparameters, including MPC parameters (horizon, lqr_iter, R_scale).

The goal is to steer the beam to a target position (minimize alignment distance) while minimizing beam spread (focusing the beam), using a reward function that emphasizes these objectives. The approach is inspired by the paper, which uses a neural policy to learn cost parameters for an MPC solver, leveraging differentiable physics simulations to enable end-to-end training.


This work incorporates and builds on ideas from:

```bibtex
@misc{romero2025actorcriticmodelpredictivecontrol,
    title        = {Actor-Critic Model Predictive Control: Differentiable Optimization meets Reinforcement Learning}, 
    author       = {Angel Romero and Elie Aljalbout and Yunlong Song and Davide Scaramuzza},
    year         = 2025,
    eprint       = {2306.09852},
    archivePrefix= {arXiv},
    primaryClass = {cs.RO},
    url          = {https://arxiv.org/abs/2306.09852}
}
```

### 📘 Cheetah Framework
This project uses the Cheetah differentiable simulation framework for accelerator lattice modeling. If you use this work, please cite the following publications as recommended by the Cheetah authors:

```bibtex
@article{kaiser2024cheetah,
    title        = {Bridging the gap between machine learning and particle accelerator physics with high-speed, differentiable simulations},
    author       = {Kaiser, Jan and Xu, Chenran and Eichler, Annika and Santamaria Garcia, Andrea},
    year         = 2024,
    month        = {May},
    journal      = {Phys. Rev. Accel. Beams},
    publisher    = {American Physical Society},
    volume       = 27,
    pages        = {054601},
    doi          = {10.1103/PhysRevAccelBeams.27.054601},
    url          = {https://link.aps.org/doi/10.1103/PhysRevAccelBeams.27.054601},
    issue        = 5,
    numpages     = 17
}

@inproceedings{stein2022accelerating,
    title        = {Accelerating Linear Beam Dynamics Simulations for Machine Learning Applications},
    author       = {Stein, Oliver and Kaiser, Jan and Eichler, Annika},
    year         = 2022,
    booktitle    = {Proceedings of the 13th International Particle Accelerator Conference}
}
```

### License
© 2025 Christian Contreras-Campana. All rights reserved.

This repository is not licensed for public or commercial use.
Any form of use, reproduction, or distribution requires explicit written permission from the author(s).
Please contact chrisjcc.physics@gmail.com for licensing inquiries.
