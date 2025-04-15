# cheetah-beam-ctrl-rl-diff-mpc
Reinforcement learning-based beam control using differentiable MPC and Cheetah accelerator simulation.

## Acknowledged and Cited Work

This project builds on the following frameworks and methods. If you use this repository, please ensure you cite the original authors as specified below.

### 🧠 Actor-Critic MPC with Differentiable Optimization

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

### 📘 Cheetah Framework
This project uses the Cheetah differentiable simulation framework for accelerator lattice modeling. If you use this work, please cite the following publications as recommended by the Cheetah authors:

### @article{kaiser2024cheetah,
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

### License
© 2025 Christian Contreras-Campana. All rights reserved.

This repository is not licensed for public or commercial use.
Any form of use, reproduction, or distribution requires explicit written permission from the author(s).
Please contact chrisjcc.physics@gmail.com for licensing inquiries.
