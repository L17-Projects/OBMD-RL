# protein_env.py  (modified to use per-env workdir)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import shutil
import os
from Q_fun import read_data, shear_to_positions, write_lammps_script, read_data

def random_rotation_matrix(rng=None):
    _rng = rng if rng is not None else np.random
    try:
        u1, u2, u3 = _rng.random(3)
    except TypeError:
        u1, u2, u3 = _rng.rand(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])
def kabsch_rotation(current_positions, target_positions):
    """
    Computes the optimal rotation matrix that aligns
    current_positions to target_positions using the Kabsch algorithm.
    """

    # Covariance matrix
    H = current_positions.T @ target_positions

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Ensure a proper rotation (determinant = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R

class ProteinEnv(gym.Env):
    def __init__(self, env_id=0, template_data='lammps/gb_start.data', lammps_exe=None, nprocs=1):
        super().__init__()
        self.env_id = int(env_id)
        self.template_data = os.path.abspath(template_data)
        if not os.path.exists(self.template_data):
            raise FileNotFoundError(f"Template data file not found: {self.template_data}")
        # create isolated workdir for this env
        self.workdir = os.path.abspath(f"lammps/env_{self.env_id}")
        os.makedirs(self.workdir, exist_ok=True)

        # copy all necessary lammps files into workdir (template data + include files)
        # copy the starting data file into the env folder with consistent name
        target_data = os.path.join(self.workdir, f"env_{self.env_id}.data")
        shutil.copy(self.template_data, target_data)

        # If your include files are needed, copy them too (adjust list as needed)
        # shutil.copy("lammps/interaction_no_ions_lj_cut.lmp", os.path.join(self.workdir, "interaction_no_ions_lj_cut.lmp"))
        # ... copy any other files referenced by in.lammps ...

        # store settings for running LAMMPS
        self.lammps_exe = lammps_exe
        self.nprocs = nprocs

        # load positions from the copied data file
        self.protein_atom_ids, positions = read_data(target_data, return_ids=True)
        self.body_points = positions.copy()

        N = self.body_points.shape[0]

        # action and observation spaces (same as before)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6 * N,), dtype=np.float32)

        # normalization / centering
        self.body_points -= np.mean(self.body_points, axis=0)
        self.normalize_scale = np.max(np.linalg.norm(self.body_points, axis=1))
        if self.normalize_scale == 0:
            self.normalize_scale = 1.0
        self.body_points /= self.normalize_scale

        self.current_positions = None
        self.target_positions = None
        self.steps = 0
        self.max_steps = 50

        # store the local data filename inside the workdir
        self.local_data = target_data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = getattr(self, 'np_random', None)

        # Always restore from pristine template to keep atom IDs consistent across episodes.
        shutil.copy(self.template_data, self.local_data)

        # sample target rotation
        R_target = random_rotation_matrix(rng=rng)

        # read the local data file (fresh copy)
        positions = read_data(self.local_data, atom_ids=self.protein_atom_ids)
        positions -= np.mean(positions, axis=0)
        positions /= self.normalize_scale
        self.body_points = positions

        # set target current and reset steps
        self.target_positions = self.body_points @ R_target.T
        self.current_positions = self.body_points.copy()
        self.steps = 0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        obs = np.concatenate([self.current_positions.flatten(), self.target_positions.flatten()])
        return np.array(obs, dtype=np.float32).flatten()

    def step(self, action):
        self.steps += 1
        a_y, a_z = (action * 0.03).tolist()

        try:
            positions = shear_to_positions(
                a_y,
                a_z,
                workdir=self.workdir,
                lammps_exe=self.lammps_exe,
                nprocs=self.nprocs,
                atom_ids=self.protein_atom_ids,
            )
            positions -= np.mean(positions, axis=0)
            positions /= self.normalize_scale

            if positions.shape != self.target_positions.shape:
                raise ValueError(
                    f"Position shape mismatch in env {self.env_id}: "
                    f"got {positions.shape}, expected {self.target_positions.shape}"
                )

            self.current_positions = positions
        except Exception as exc:
            info = {
                "mean_distance": float("inf"),
                "error": str(exc),
                "env_id": self.env_id,
            }
            obs = self._get_obs()
            return obs, -np.pi, True, True, info

        dists = np.linalg.norm(self.current_positions - self.target_positions, axis=1)
        R = kabsch_rotation(self.current_positions, self.target_positions)
        angle_cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
        angle = np.arccos(angle_cos)

        reward = -angle
        terminated = False
        truncated = self.steps >= self.max_steps
        info = {"mean_distance": float(np.mean(dists))}

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
