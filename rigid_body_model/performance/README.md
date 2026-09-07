# Measuring rigid-body performance

[Rigid-body model](../README.md)

`measure_performance.py` evaluates saved policies over independent random initial and target orientations. For each checkpoint it records the initial and final Kabsch angles and the deterministic action at every rollout step. Success in the manuscript is the fraction of final angles below $10^\circ$.

The checked-in entry point uses ubiquitin (`N = 166`), 100 tests, and 30 actions per test. The manuscript's final policy analysis uses larger ensembles where stated; increase `n_tests` before producing publication statistics.

`rigid_body_perturbed_env.py` is the serial evaluation environment. `rigid_body_perturbed_env_par.py` and `SACpar.py` are parallel-training copies retained so Stable-Baselines3 can reconstruct policies that reference the custom SO(2) replay buffer. The three `.npy` coordinate files provide the same GB1, ubiquitin, and DHFR shapes used in training.

Before running, copy or link the desired checkpoints into `checkpoints_SAC_perturbed/`, or edit the call at the bottom of `measure_performance.py` to use another directory. Then run:

```bash
python measure_performance.py
```

The checkpoint directory is not included in this snapshot. The saved arrays used for the manuscript's DDPG/TD3/SAC and replay-strategy comparisons are under `../plots/fig2/` and `../plots/fig3/`.

For each requested variant, outputs are:

| Output | Contents |
| --- | --- |
| `checkpoint_results_*.csv` | Checkpoint step, mean initial/final angle, change in mean angle, final-angle standard deviation, and model path. Angles are in radians. |
| `all_results_*.npy` | Array indexed by checkpoint and test, with `[initial_angle, final_angle]`. |
| `actions_step_<step>.npy` | Pickled dictionary mapping action index to the actions collected across tests. |

Angles are stored in radians; convert them to degrees only for reporting or plotting. The `improvement` column is `mean_final - mean_init`, so successful control generally gives a negative value. Action files do not include a variant label and can overwrite one another when several checkpoint families are evaluated in the same directory.
