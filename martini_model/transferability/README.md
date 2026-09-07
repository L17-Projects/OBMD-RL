# Evaluating a compatible policy on ubiquitin

[Main guide](../README.md) · [Evaluation interface and outputs](../performance/README.md)

This workflow applies a saved policy to ubiquitin using the same deterministic evaluation interface as `performance/`. `protein_env_prll_perf.py` selects `ubq_start.data`; `Q_funalgo.py` uses protein types below 23 and OBMD solvent type 23. `models/` contains checkpoints and saved results; `rltest_ubq.ipynb` explores angle and action histories.

This directory tests transfer across the checked-in protein/input configuration. It is not, by itself, the central transfer experiment described in the manuscript. That experiment applies a rigid-body policy trained for GB1 directly to the GB1 molecular environment without additional optimization. The corresponding saved rigid-versus-molecular comparison is under `../plots/fig6/`; the paper reports about $95\%$ success within $20^\circ$ for the transferred rigid-body policy and about $99\%$ for the directly trained molecular policy at $4\times10^5$ training steps.

The `lammps/` folder is currently empty. Supply a mutually consistent `ubq_start.data`, `interaction_no_ions_lj_cut.lmp`, and `exclude_groups.lmp` before running. Verify that the selected checkpoint accepts the resulting `6N` observation and bead representation; this code does not remap observations between proteins.

After supplying inputs and configuring LAMMPS, run from this folder:

```bash
bash job.sh
```

`run_perf_v2.sh` is the Slurm array launcher: 64 tasks × 10 rollouts per checkpoint. `job.sh` wraps submission and cleanup. The local `SAC_prll.py` copy imports an absent `protein_env_prll`; train using `../training/`.

