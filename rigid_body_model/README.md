# Rigid-body orientation-control model

This reduced model isolates the geometry of protein orientation control from solvent dynamics, thermal noise, and molecular deformation. A protein is represented by a rigid set of centered bead coordinates. Independent rotations sampled uniformly from SO(3) define the initial and target orientations at the start of each episode.

As in the molecular OBMD-RL model, the goal-conditioned state concatenates the current and target coordinates and has size $6N$. The two-component action $\mathbf{a}=(a_y,a_z)\in[-1,1]^2$ parameterizes a bounded rotation in the transverse control plane. In this implementation,

$$
\theta=\theta_{\max}\sqrt{a_y^2+a_z^2},
\qquad
\hat{\mathbf{n}}=\frac{(0,-a_z,a_y)}{\sqrt{a_y^2+a_z^2}},
$$

where $\theta_{\max}=\pi/18=10^\circ$ by default. The body is rotated through $\theta$ about $\hat{\mathbf{n}}$. This is the code's axis-angle realization of the two bounded rotation controls described in the manuscript. It is the kinematic analogue of a transverse shear action: the rotation axis is perpendicular to both the $x$ direction and the transverse action vector.

The reward is $r_t=-\theta_t$, where $\theta_t$ is the Kabsch orientation error in radians. Training episodes stop when the mean point displacement falls below `0.01` or after 50 actions. Because the body is exactly rigid in the checked-in configuration, the model contains no physical time scale and no conformational degrees of freedom.

## Protein representations

The supplied point clouds are:

| `N` | Body |
| ---: | --- |
| 4 or fewer | Built-in tetrahedron-like toy body. |
| 131 | GB1 coordinates from `gb.npy`. |
| 166 | Ubiquitin coordinates from `protein.npy`; the default training body. |
| 369 | Dihydrofolate reductase coordinates from `dfr.npy`. |
| Other | A newly generated random point cloud. |

Coordinates are centered, making the state insensitive to translation, and protein point clouds are normalized by their mean distance from the center. The three proteins match the benchmark systems in the manuscript.

## SO(2) replay augmentation

Rotating both coordinate sets and the action by the same angle $\phi$ in the $yz$ plane produces an equivalent transition. The full-batch replay buffer applies

$$
(\mathbf{s}_t,\mathbf{a}_t,r_t,\mathbf{s}_{t+1})
\longrightarrow
(R_\phi\mathbf{s}_t,R_\phi\mathbf{a}_t,r_t,R_\phi\mathbf{s}_{t+1}),
$$

with $\phi$ sampled uniformly from $[0,2\pi)$. The manuscript compares this strategy with unaugmented, half-batch, and doubled-sample variants. At $4\times10^5$ training steps, it reports $71\%$ success within $10^\circ$ without augmentation and $100\%$ for each augmented strategy.

## Folders

| Folder | Purpose |
| --- | --- |
| [training](training/README.md) | Train SAC with parallel environments and full-batch SO(2) replay augmentation. |
| [performance](performance/README.md) | Evaluate checkpoints over random initial and target orientations and save angle/action statistics. |
| [`plots/fig2`](plots/fig2/) | Saved DDPG, TD3, and SAC comparison used for the algorithm benchmark. |
| [`plots/fig3`](plots/fig3/) | Saved comparison of the four replay-sampling strategies. |

The manuscript finds that SAC offers the best balance of learning speed, stability, and final performance among DDPG, TD3, and SAC, and therefore uses SAC for the molecular simulations. Run scripts from their own folders because coordinate and output paths are relative to the working directory. Required Python packages are NumPy, PyTorch, Gymnasium, Stable-Baselines3, pandas, and Matplotlib; the notebooks also require Jupyter. No environment or version lockfile is included.
