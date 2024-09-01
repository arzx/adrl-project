# Advanced Deep Reinforcement Learning
This project aims to combine the exploration strategy offered by Random Network Distillation (RND) with the robustness of Proximal Policy Optimization (PPO). As previous attempts improved smaller state-space environments, this project is thought to solve the milestones from Crafter environment which is a 2D abstraction of Minecraft. The main challenge in this project is the utilization of intrinsic rewards while the base environment has incorporated milestones which are essential for the reward function and solving the environment. 

## Results

## Installation
### (Recommended) Setup new clean environment
Use a conda package manager.

#### Conda
Subsequently run these commands, following the prompted runtime instructions:
```bash
conda create -n crafter python=3.12.15
conda activate crafter
pip install -r requirements.txt
```

## Rewards in Crafter
Reward: The sparse reward is +1 for unlocking an achievement during the episode and -0.1 or +0.1 for lost or regenerated health points. Results should be reported not as reward but as success rates and score.

Success rates: The success rates of the 22 achievemnts are computed as the percentage across all training episodes in which the achievement was unlocked, allowing insights into the ability spectrum of an agent.

## Used Hardware
| Compute/Evaluation Infrastructure    |                                      |
|:-------------------------------------|--------------------------------------|
| Device                               | MacBook Pro M3 Pro 14-Inch                  |
| CPU                                  | M3 Pro |
| GPU                                  | -                                    |
| TPU                                  | -                                    |
| RAM                                  | 18 GB RAM                       |
| OS                                   | Sonoma 14.5                        |
| Python Version                       | 3.10.14                               |

## Further Work
Apply this in Jax for faster inference times. There is also a Jax-based Crafter equivalent which is called Craftax [3].
## Sources
[1] Nugroho, W. (2021). Reinforcement Learning PPO RND [Source code]. GitHub. https://github.com/wisnunugroho21/reinforcement_learning_ppo_rnd
[2] Hafner, D. (2021). Benchmarking the Spectrum of Agent Capabilities. arXiv preprint:2109.06780
[3] Matthews, M. et al. (2024). Craftax: A Lightning-Fast Benchmark for Open-End Reinforcement Learning. International Conference on Machine Learning (ICML)