# OBMD-RL with coarse-grained Martini proteins

This model couples Soft Actor-Critic (SAC) reinforcement learning to explicit-solvent Martini 3 simulations. The protein, Martini water, and elastic network evolve in LAMMPS, while `fix obmd` imposes momentum fluxes in open-boundary buffer regions. The policy controls the solvent; protein rotation emerges from the coupled hydrodynamic and molecular dynamics.

## State, action, and reward

The goal-conditioned state concatenates the centered, normalized coordinates of the current and target protein conformations:

$$
\mathbf{s}_t=\left(\mathbf{x}_t,\mathbf{x}^{\mathrm{target}}\right)\in\mathbb{R}^{6N}.
$$

For a normalized action $\mathbf{a}_t=(a_y,a_z)$ with each component in $[-1,1]$, the code applies

$$
\left(P_{xy},P_{xz}\right)=0.03\left(a_y,a_z\right).
$$

The reward is $r_t=-\theta_t$, where $\theta_t$ is the Kabsch rotation angle between the current and target structures in radians. Centering removes global translation from this objective. Each transition propagates 5,000 MD steps: shear is applied for 2,000 steps, followed by 3,000 steps without shear to dissipate residual solvent flow.

## Physical picture

The simulation cell is open in the $x$ direction and periodic in $y$ and $z$. The components $P_{xy}$ and $P_{xz}$ describe tangential loading in the $y$ and $z$ directions, respectively, across surfaces normal to $x$. 

The OBMD buffers maintain the open-boundary fluid and impose the transverse momentum flux. Momentum transferred from the solvent produces conformation-dependent hydrodynamic forces and torque on the protein. A weak harmonic restraint keeps the protein center of mass near the box center without directly constraining its rotation.

The manuscript reports the following physical parameters:

| Parameter | Value |
| --- | ---: |
| Temperature | 300 K |
| Integration timestep | 3 fs |
| Shear / relaxation duration | 2,000 / 3,000 steps |
| Maximum $|P_{xy}|$ and $|P_{xz}|$ | $0.03\ \mathrm{kcal\,mol^{-1}\,\mathring{A}^{-3}}$ |
| DPD dissipation parameter | $4.8\ \mathrm{ps\,kcal\,mol^{-1}\,\mathring{A}^{-2}}$ |
| DPD cutoff | $10\ \mathring{A}$ |
| OBMD buffer relaxation time | 30 fs |
| OBMD buffer mass scale | 0.95 |
| Center-of-mass spring stiffness | $10.0\ \mathrm{kcal\,mol^{-1}}$ |
| Normal pressure $P_{xx}$ | 0 |


## Symmetry and reported behavior

Training uses full-batch SO(2) replay augmentation: current coordinates, target coordinates, next coordinates, and the two stress components are rotated consistently about the $x$ axis, while the scalar reward is unchanged. Unlike the exact symmetry of the rigid-body model, this equivariance is approximate here because a finite periodic cell introduces a preferred reference frame.

The paper reports success rates above $90\%$ within $20^\circ$ for GB1, UBQ, and DHFR. For policies trained for $4\times10^5$ environment steps, the final angular mismatches are $8.6^\circ\pm6.2^\circ$, $12.1^\circ\pm4.9^\circ$, and $7.2^\circ\pm9.7^\circ$, respectively. Early actions tend to have large magnitude; after roughly 10–11 control steps, the policy shifts toward smaller corrective actions. A fixed-target DHFR example reduces a $90^\circ$ mismatch to about $5^\circ$ while keeping the reported backbone RMSD within approximately $2\ \mathring{A}$.

## Repository workflows

| Folder | Purpose |
| --- | --- |
| [training](training/README.md) | Train an SO(2)-augmented SAC policy directly on GB1 with parallel LAMMPS environments. |
| [performance](performance/README.md) | Evaluate checkpoints on DHFR and save orientation and action histories. |
| [transferability](transferability/README.md) | Apply compatible checkpoints to ubiquitin; this is distinct from the paper's rigid-to-molecular GB1 transfer comparison. |
| [example protocol](example_protocol/README.md) | Reproduce the fixed $90^\circ$ DHFR target and analyze orientation, RMSD, and radius of gyration. |
| [plots](plots/README.md) | Inspect the saved data used for the three-protein and rigid-to-molecular comparisons. |

