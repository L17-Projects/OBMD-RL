# Fixed-target DHFR shear protocol

[Main guide](../README.md) · [Evaluation interface and outputs](../performance/README.md)

This workflow underlies the manuscript's representative DHFR control protocol. It records shear actions, Kabsch orientation errors, and the protein trajectory for a target obtained by rotating the initial conformation by $90^\circ$ about the $(1,1,1)$ axis. `algorithm_performance.py`, `protein_env_prll_perf.py`, and `Q_funalgo.py` have the same roles as in `performance/`. `lammps/` contains the starting data and force-field includes; `models/` contains a saved policy and evaluation results. The local `SAC_prll.py` is a training copy with an absent environment import; use `../training/` for training.

## Choose and run a rotation

The supplied environment currently samples **random targets**. To reproduce the manuscript protocol, replace the active `R_target = random_rotation_matrix(rng=rng)` line in `ProteinEnv.reset()` in `protein_env_prll_perf.py` with:

```python
R_target = rotate_by_angle(np.array([1.0, 1.0, 1.0]), np.pi / 2)
```

`rotate_by_angle` normalizes the axis; the angle is in radians. The target is `body_points @ R_target.T`, relative to the centered current structure at reset. This choice matches the default target in `kab.py`; keep both settings consistent for other rotations.

After configuring the executable path, run one protocol from this folder:

```bash
bash job.sh
```

## Analyze a protocol

Set each script's path constants before running `python SCRIPT.py` from this folder:

| Script | Analysis / requirements |
| --- | --- |
| `gyr.py` | Radius of gyration from `lammps/model_400000/eval_*/protein.lammpstrj`; writes under `radius_of_gyration/`. |
| `kab.py` | Rotation error relative to the configured rotated initial structure; writes `kabsch_angles_all.csv` under the model directory. |
| `rms.py` | RMSD after centering and Kabsch alignment to the first trajectory frame; writes `rmsd_all.csv`. |
| `rms_byid.py` | RMSD for selected atom IDs against a fixed data-file reference; needs `resid_all.npy` and its reference path corrected to `lammps/dfr_start.data`. |
| `rms_backbone.py` | Selected-atom RMSD against `lammps/dfr_start.data`; also needs `resid_all.npy`; writes `rmsd_all_equi.csv` under the model directory. |
| `rltest_ubq.ipynb` | Inspect saved angles and actions; its inherited name does not change this folder's DHFR inputs. Update the referenced `best_model.zip` to an available checkpoint. |

`resid_all.npy` is not supplied. Select IDs appropriate to the intended backbone/beads before using those analyses. RMSD and radius of gyration characterize structural deformation; the Kabsch angle characterizes orientation. The manuscript reports that this protocol reduces the mismatch from $90^\circ$ to approximately $5^\circ$, while the backbone RMSD remains within roughly $2\ \mathring{A}$ of the reference behavior. Those are reported results, not assertions checked by these scripts at run time.
