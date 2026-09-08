# Active Nanoscale Control of Protein Orientation via Symmetry-Informed Reinforcement Learning

[![MPI Supported](https://img.shields.io/badge/MPI-supported-brightgreen?logo=mpi)](https://www.open-mpi.org/) [![SLURM Supported](https://img.shields.io/badge/SLURM-supported-blue?logo=slurm)](https://slurm.schedmd.com/) [![LAMMPS Required](https://img.shields.io/badge/LAMMPS-required-orange)](https://lammps.sandia.gov/) [![Python 3](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-supported-EE4C2C?logo=pytorch)](https://pytorch.org/) [![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-orange?logo=jupyter)](https://jupyter.org/)


This repository contains the code and saved analysis for the OBMD-RL framework. The method uses goal-conditioned reinforcement learning to steer a protein from its current orientation to a prescribed target orientation. In the molecular model, the controller acts on the surrounding solvent through two transverse shear stresses rather than rotating the protein directly.

The state combines the centered bead coordinates of the current and target structures,

$$
\mathbf{s}_t = \left(\mathbf{x}_t,\mathbf{x}^{\mathrm{target}}\right),
$$

and the reward is the negative orientational mismatch,

$$
r_t=-\theta_t.
$$

The angle $\theta_t$ is obtained from the optimal proper rotation $R_t$ returned by Kabsch alignment:

$$
\theta_t=\arccos\!\left[\frac{\operatorname{tr}(R_t)-1}{2}\right].
$$

Centering makes this objective invariant to global translation. Random initial and target rotations make the policy goal-conditioned over orientations rather than specialized to one target.


## Symmetry-informed learning

The control problem is equivariant under rotations $R_\phi$ about the $x$ axis, or equivalently within the transverse $yz$ plane. The replay-buffer augmentation transforms a sampled transition as

$$
(\mathbf{s}_t,\mathbf{a}_t,r_t,\mathbf{s}_{t+1})
\longrightarrow
(R_\phi\mathbf{s}_t,R_\phi\mathbf{a}_t,r_t,R_\phi\mathbf{s}_{t+1}).
$$

The paper compares unmodified, full-batch, half-batch, and doubled-sample replay strategies and selects full-batch SO(2) augmentation for the molecular simulations. 

## Results represented by this repository

In the rigid-body benchmarks, SAC converges faster than TD3 and DDPG. At $4\times10^5$ training steps, the unaugmented replay strategy reaches a reported $71\%$ success rate within $10^\circ$, whereas all three SO(2)-augmented strategies reach $100\%$.

For directly trained Martini policies, the paper reports success rates above $90\%$ within $20^\circ$ for all three proteins. At $4\times10^5$ training steps, the reported final angular mismatches are $8.6^\circ\pm6.2^\circ$ for GB1, $12.1^\circ\pm4.9^\circ$ for UBQ, and $7.2^\circ\pm9.7^\circ$ for DHFR. A GB1 rigid-body policy transferred to the molecular environment without additional training reaches about $95\%$ success within $20^\circ$, compared with about $99\%$ for a policy trained directly in molecular simulation.

## Software and execution notes

The Python workflows use NumPy, PyTorch, Gymnasium, Stable-Baselines3, pandas, Matplotlib, and Jupyter. Molecular simulations additionally require MPI and a LAMMPS build containing the project-specific `fix obmd` implementation and all force-field styles used by the generated inputs. 