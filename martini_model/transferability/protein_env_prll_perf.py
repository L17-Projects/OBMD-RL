import gymnasium as gym
from gymnasium import spaces
import numpy as np
import shutil
import os
from Q_funalgo import read_data, shear_to_positions, write_lammps_script

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

def rotate_by_angle(axis, angle):
    """
    Returns a rotation matrix that rotates by a given angle around a specified axis.
    """
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1 - cos_angle

    x, y, z = axis
    R = np.array([
        [cos_angle + x**2 * one_minus_cos, x*y*one_minus_cos - z*sin_angle, x*z*one_minus_cos + y*sin_angle],
        [y*x*one_minus_cos + z*sin_angle, cos_angle + y**2 * one_minus_cos, y*z*one_minus_cos - x*sin_angle],
        [z*x*one_minus_cos - y*sin_angle, z*y*one_minus_cos + x*sin_angle, cos_angle + z**2 * one_minus_cos]
    ])
    return R

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
    def __init__(self, workdir, template_data='lammps/ubq_start.data', lammps_exe=None, nprocs=1):
        super().__init__()
        #self.env_id = int(env_id)
        # create isolated workdir for this env
        
        positions = read_data(f"{template_data}")

        self.body_points = positions.copy()

        self.workdir = os.path.abspath(workdir)

        print(f"Initialized ProteinEnv with workdir: {self.workdir}")
        
        #kopira lammps/ubq_start.data v lammps/eval_id/rotated.data
        shutil.copy(f"{template_data}", f"{self.workdir}/rotated.data")

        #os.makedirs(self.workdir, exist_ok=True)

        # copy all necessary lammps files into workdir (template data + include files)
        # copy the starting data file into the env folder with consistent name
        #target_data = os.path.join(self.workdir, f"env_{self.env_id}.data")
        #shutil.copy(template_data, target_data)

        # If your include files are needed, copy them too (adjust list as needed)
        # shutil.copy("lammps/interaction_no_ions_lj_cut.lmp", os.path.join(self.workdir, "interaction_no_ions_lj_cut.lmp"))
        # ... copy any other files referenced by in.lammps ...

        # store settings for running LAMMPS
        self.lammps_exe = lammps_exe
        self.nprocs = nprocs

        # load positions from the copied data file
        #positions = read_data(target_data)
        #self.body_points = positions.copy()

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
        self.local_data = self.workdir +  '/rotated.data'
        self.local_data_template = self.workdir +  '/../ubq_start.data'

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = getattr(self, 'np_random', None)

        # sample target rotation
        #R_target = rotate_by_angle(np.array([0, 1, 0]), np.pi/2) # here theta variable) #random_rotation_matrix(rng=rng)

        R_target = random_rotation_matrix(rng=rng)

        # read the local data file (fresh copy)
        positions = read_data(self.local_data)
        positions -= np.mean(positions, axis=0)
        positions /= self.normalize_scale
        self.body_points = positions

        # set target current and reset steps
        self.target_positions = self.body_points @ R_target.T
        self.current_positions = self.body_points.copy()
        self.steps = 0

        obs = self._get_obs()
        return obs, {}

    def _get_unnormalized_target(self):
        return self.target_positions * self.normalize_scale

    def _get_obs(self):
        obs = np.concatenate([self.current_positions.flatten(), self.target_positions.flatten()])
        return np.array(obs, dtype=np.float32).flatten()

    def step(self, action):
        self.steps += 1
        a_y, a_z = (action * 0.03).tolist()

        # call Q_fun.shear_to_positions with workdir and executable settings
        # shear_to_positions will write in.lammps to self.workdir and run LAMMPS there
        positions = shear_to_positions(a_y, a_z, workdir=self.workdir, lammps_exe=self.lammps_exe, nprocs=self.nprocs)

        # recenter & normalize
        positions -= np.mean(positions, axis=0)
        positions /= self.normalize_scale
        self.current_positions = positions

        dists = np.linalg.norm(self.current_positions - self.target_positions, axis=1)
        #reward = -np.mean(dists)
        R = kabsch_rotation(self.current_positions, self.target_positions)
        angle = np.arccos((np.trace(R) - 1) / 2)

        reward = -angle
        terminated = False
        truncated = self.steps >= self.max_steps
        info = {"mean_distance": float(np.mean(dists))}

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
