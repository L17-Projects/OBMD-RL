# Active Nanoscale Control of Protein Orientation via Symmetry-Informed Reinforcement Learning

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
\theta_t=\arccos\left[\frac{\mathrm{tr}(R_t)-1}{2}\right].
$$

Centering makes this objective invariant to global translation. Random initial and target rotations make the policy goal-conditioned over orientations rather than specialized to one target.

## Models and workflows

| Component | Description |
| --- | --- |
| [Rigid-body model](rigid_body_model/README.md) | A computationally inexpensive, noise-free point-cloud model used to compare DDPG, TD3, and SAC and to study symmetry-informed replay sampling. Actions apply bounded rotations in the transverse control plane. |
| [Martini/OBMD model](martini_model/README.md) | Explicit-solvent coarse-grained Martini 3 simulations driven by the OBMD implementation in LAMMPS. Actions set $P_{xy}$ and $P_{xz}$, and protein rotation emerges from solvent-mediated hydrodynamic torque. |

The manuscript studies three globular proteins with different sizes and shapes: protein G (GB1), ubiquitin (UBQ), and dihydrofolate reductase (DHFR). The repository includes training, checkpoint evaluation, fixed-target trajectory analysis, transfer tests, and plotting notebooks; consult the model-specific guides because scripts use paths relative to their own directories.

## Symmetry-informed learning

The control problem is equivariant under rotations $R_\phi$ about the $x$ axis, or equivalently within the transverse $yz$ plane. The replay-buffer augmentation transforms a sampled transition as

$$
(\mathbf{s}_t,\mathbf{a}_t,r_t,\mathbf{s}_{t+1})
\longrightarrow
(R_\phi\mathbf{s}_t,R_\phi\mathbf{a}_t,r_t,R_\phi\mathbf{s}_{t+1}).
$$

The reward remains unchanged because it is a scalar orientational error. The paper compares unmodified, full-batch, half-batch, and doubled-sample replay strategies and selects full-batch SO(2) augmentation for the molecular simulations. This symmetry is exact in the rigid-body environment and approximate in the molecular model because the finite periodic cell supplies a preferred reference frame.

## Results represented by this repository

In the rigid-body benchmarks, SAC converges faster than TD3 and DDPG. At $4\times10^5$ training steps, the unaugmented replay strategy reaches a reported $71\%$ success rate within $10^\circ$, whereas all three SO(2)-augmented strategies reach $100\%$.

For directly trained Martini policies, the paper reports success rates above $90\%$ within $20^\circ$ for all three proteins. At $4\times10^5$ training steps, the reported final angular mismatches are $8.6^\circ\pm6.2^\circ$ for GB1, $12.1^\circ\pm4.9^\circ$ for UBQ, and $7.2^\circ\pm9.7^\circ$ for DHFR. A GB1 rigid-body policy transferred to the molecular environment without additional training reaches about $95\%$ success within $20^\circ$, compared with about $99\%$ for a policy trained directly in molecular simulation.

These values summarize the manuscript analyses and are not recomputed automatically by the README workflows.

## Software and execution notes

The Python workflows use NumPy, PyTorch, Gymnasium, Stable-Baselines3, pandas, Matplotlib, and Jupyter. Molecular simulations additionally require MPI and a LAMMPS build containing the project-specific `fix obmd` implementation and all force-field styles used by the generated inputs. No version-locked environment is supplied.

