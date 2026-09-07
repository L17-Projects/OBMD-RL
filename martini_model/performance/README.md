# Policy performance

[Main guide](../README.md)

`algorithm_performance.py` measures orientation accuracy across random targets using deterministic SAC or TD3 inference. Each rollout records the initial Kabsch angle and the angles after 30 control actions. The paper defines molecular-model success as a final angular mismatch below $20^\circ$.

`protein_env_prll_perf.py` defines the environment; `Q_funalgo.py` generates and runs OBMD simulations. `lammps/` supplies DHFR data, interactions, and exclusions. `models/` holds checkpoints and saved evaluation arrays. The local `SAC_prll.py` is a training-code copy importing `protein_env_prll`, which is not supplied in this folder; use the training entry point for training.

After setting the executable path and checking checkpoint compatibility, run one rollout from this folder (choose an unused output index if preserving existing results):

```bash
bash job.sh
```

Arguments are `MODEL_NAME MODEL_TYPE START END WORKDIR`; the model is loaded from `models/MODEL_NAME.zip`. `END - START` is the rollout count; `START` labels the output files, **not a random seed**. `run_job.sh` reproduces the manuscript-scale ensemble of 64 Slurm tasks with 10 rollouts each (640 episodes per checkpoint).

For policies trained for $4\times10^5$ environment steps, the manuscript reports final mismatches of $8.6^\circ\pm6.2^\circ$ for GB1, $12.1^\circ\pm4.9^\circ$ for UBQ, and $7.2^\circ\pm9.7^\circ$ for DHFR, with success rates above $90\%$ for all three proteins. This folder's default inputs evaluate DHFR; the saved cross-protein comparison is under `../plots/fig5/`.

## Outputs

All arrays use the suffix `<model_name>_<start_index>.npy` in `models/`:

| Prefix | Contents |
| --- | --- |
| `results_initial_` / `results_final_` | Initial/final Kabsch angles in radians, despite old “dot product” variable names. |
| `angles_over_episodes_` | One row per rollout, normally 31 angles including the initial value. |
| `action_dict_` | Pickled dictionary: action-step index → list of normalized action pairs. Multiply by `0.03` for applied stresses. |

The work directory contains `rotated.data`, generated `in.lammps`, appended `protein.lammpstrj`, and `start_*.xyz` / `target_*.xyz`. Evaluation resets sample a new target from the current configuration; they do not restore the pristine data between rollouts. Use separate fresh work directories for independent starts.
