# Training the rigid-body SAC policy

[Rigid-body model](../README.md)

`rigid_body_perturbed_env_par.py` implements the Gymnasium environment and analytical action-to-rotation map. `SACpar.py` trains the Soft Actor-Critic (SAC) policy used by the symmetry-informed workflow.

The checked-in configuration follows the principal manuscript settings:

| Parameter | Value |
| --- | ---: |
| Training body | Ubiquitin (`N = 166`) |
| Parallel training environments | 4 |
| Parallel evaluation environments | 4 |
| Hidden layers | `[512, 512]` |
| Learning rate | $3\times10^{-4}$ |
| Replay-buffer capacity | $10^6$ transitions |
| Batch size | 512 |
| Episode limit | 50 actions |
| Requested training length | $5\times10^5$ transitions |

`SO2ReplayBuffer` implements the paper's full-batch augmentation (Sampling II). Every sampled current state, target state, next state, and two-component action is rotated consistently by a random angle about the $x$ axis; scalar rewards and terminal flags are unchanged. This enforces exact SO(2) equivariance in the rigid-body environment.

From this folder, create the output directories and run:

```bash
python SACpar.py
```

Scheduled policies are written as `checkpoints_SAC_perturbed/model_<step>.zip`. Evaluation histories are stored below `logs_SAC_perturbed/`, the best policy below `logs_SAC_long/`, and `timestep.txt` records transition count against elapsed time. These output directories must exist before training.

Change `N` in `SACpar.py` to select GB1 (`131`), ubiquitin (`166`), or DHFR (`369`). Keep `N_atoms` in the replay-buffer arguments equal to the environment's `N`, or replay observations cannot be reshaped correctly. Despite the filenames, the current perturbation scale is `0.0`, so the points remain exactly rigid.
