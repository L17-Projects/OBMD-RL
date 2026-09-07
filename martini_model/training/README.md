# Training the molecular OBMD-RL policy

[Main guide](../README.md)

- `SAC_prll.py`: training entry point; 32 training and 32 evaluation subprocess environments, a `[512, 512]` MLP, and 500,000 requested transitions. `ProteinSO2ReplayBuffer` implements full-batch SO(2) augmentation by rotating coordinates and shear actions consistently about the $x$ axis.
- `protein_env_prll.py`: Gymnasium environment, random target rotations, template restoration on reset, and 50-action episode limit. Protein atom IDs are retained across MD steps.
- `Q_fun.py`: reads coordinates, generates OBMD input, and launches MPI LAMMPS.
- `lammps/`: GB1 starting configuration (`gb_start.data`), force-field coefficients (`interactions.lmp`), and exclusions (`exclude_groups.lmp`). Generated `env_<id>/` directories isolate simulation state; evaluation IDs start at 1000.
- `rltest.ipynb`: examines training evaluation history.

The principal reinforcement-learning parameters match the manuscript: learning rate $3\times10^{-4}$, replay capacity $10^6$, batch size 512, two hidden layers of 512 units, and a 50-action episode limit. The code requests 32 gradient steps after each vectorized step, corresponding to one update per transition collected from the 32 environments. SAC uses its default automatic entropy-coefficient optimization and bounded `tanh` action output.

At every replay sample, the current state, target state, action, and next state are transformed by the same random $yz$-plane rotation. Rewards and terminal flags remain unchanged. This is Sampling II—the full-batch strategy selected in the manuscript after the rigid-body comparison.


```bash
sbatch run.sh
```

`checkpoints_SAC/model_<step>.zip` contains scheduled checkpoints; `logs_SAC_protein/` holds the best model and `evaluations.npz`; `timestep.txt` records transitions against elapsed seconds. Copy selected checkpoints into an evaluation folder's `models/` directory.
