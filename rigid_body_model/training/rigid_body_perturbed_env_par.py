import gymnasium as gym
from gymnasium import spaces
import numpy as np

# matrix is used for selcting target
def random_rotation_matrix(rng=None):
    """Generate a uniform random rotation matrix.

    If ``rng`` is provided it must implement ``rng.random(n)`` (like numpy's Generator
    or the global ``np.random``), otherwise the global ``np.random`` is used.
    """
    # Method: random unit quaternion
    _rng = rng if rng is not None else np.random
    # rng.random(3) for both Generator and numpy.random compatible objects
    try:
        u1, u2, u3 = _rng.random(3)
    except TypeError:
        # fallback for older numpy.random.RandomState which uses rand
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

# calculate motion of rigid body for given actions
# anlge of rotation is selected with rotation scale (angle in radians)
def action_to_matrix(a_y, a_z, rotation_scale):
    """Example: convert action values into rotation matrix around y and z."""
    angle = (a_y**2 + a_z**2)**0.5 * rotation_scale
    vel = np.array([0, a_y, a_z])
    
    if np.linalg.norm(vel) < 1e-5:
        return np.eye(3)
    
    # rotate vel[1:] (2d vector) for 90 deg counterclockwise to get axis
    axis = np.array([0, -vel[2], vel[1]])
    axis /= np.linalg.norm(axis)
    
    R = np.array([[np.cos(angle) + axis[0]**2 * (1 - np.cos(angle)),
                    axis[0]*axis[1]*(1 - np.cos(angle)) - axis[2]*np.sin(angle),
                    axis[0]*axis[2]*(1 - np.cos(angle)) + axis[1]*np.sin(angle)],
                    [axis[1]*axis[0]*(1 - np.cos(angle)) + axis[2]*np.sin(angle),
                    np.cos(angle) + axis[1]**2 * (1 - np.cos(angle)),
                    axis[1]*axis[2]*(1 - np.cos(angle)) - axis[0]*np.sin(angle)],
                    [axis[2]*axis[0]*(1 - np.cos(angle)) - axis[1]*np.sin(angle),
                    axis[2]*axis[1]*(1 - np.cos(angle)) + axis[0]*np.sin(angle),
                    np.cos(angle) + axis[2]**2 * (1 - np.cos(angle))]])

    return R

def generate_body_points(N=4, max_radius=1.0, seed=None):
    """
    Generate N 3D points roughly centered around origin.
    Most points will be within max_radius from origin.

    max_radius : float
        Desired approximate radius of the cluster.
    seed : None - random
    """
    rng = np.random.default_rng(seed)

    # Start with Gaussian distribution around origin
    points = rng.normal(size=(N, 3))

    # Normalize each point length to limit outliers
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    scaled = points / norms

    # Draw random radii (more dense near center)
    radii = rng.uniform(0, 1, size=(N, 1)) ** (1/3)
    points = scaled * radii * max_radius

    # Center exactly around origin (optional)
    points -= np.mean(points, axis=0)

    return points


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

class RigidBodyRotationEnv(gym.Env):
    """Rigid body environment: rotate a simple 3D body to match target orientation."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_id = 0, N=4, max_angle=np.pi/18):
        super().__init__()
        self.env_id = int(env_id)
        self.N = N
        self.max_angle = max_angle

        # Continuous actions: rotation around Y and Z
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation: positions of current and target body (6N)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6 * N,), dtype=np.float32
        )

        # Example rigid body (tetrahedron-like)
        if N <= 4:
            self.body_points = np.array([
                [1, 0, -1/np.sqrt(2)],
                [-1, 0, -1/np.sqrt(2)],
                [0, 1, 1/np.sqrt(2)],
                [0, -1, 1/np.sqrt(2)]
            ])[:N]
        elif N == 166: # for protein ubiquitin
            positions = np.load("protein.npy")
            positions -= np.mean(positions, axis=0)
            positions /= np.mean(np.linalg.norm(positions, axis=1))  # normalize size
            self.body_points = positions
        elif N == 131: # for protein GB1
            positions = np.load("gb.npy")
            positions -= np.mean(positions, axis=0)
            positions /= np.mean(np.linalg.norm(positions, axis=1))  # normalize size
            self.body_points = positions
        elif N == 369: # for protein dihidrofolate reductase
            positions = np.load("dfr.npy")
            positions -= np.mean(positions, axis=0)
            positions /= np.mean(np.linalg.norm(positions, axis=1))  # normalize size
            self.body_points = positions
        else:
            # import base from npy file
            self.body_points = generate_body_points(N=N, max_radius=1.0)

        self.body_points -= np.mean(self.body_points, axis=0)

        self.current_positions = None
        self.target_positions = None
        self.steps = 0
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Use the environment's RNG (set by super().reset(seed=...)) when available
        rng = getattr(self, 'np_random', None)

        # Randomly set initial and target orientations using the env RNG so that
        # env.reset(seed=...) controls reproducibility properly.
        R_target = random_rotation_matrix(rng=rng)
        R_init = random_rotation_matrix(rng=rng)

        self.target_positions = self.body_points @ R_target.T
        self.current_positions = self.body_points @ R_init.T
        # make small perturbation to initial positions
        # find size of clustrer of points
        cluster_size = np.mean(np.linalg.norm(self.current_positions - np.mean(self.current_positions, axis=0), axis=1))
        perturbation_scale = cluster_size * 0.0  # select level of noise, x% of cluster size
        # add gaussian noise on a perturbation scale
        noise = np.random.normal(scale=perturbation_scale, size=self.current_positions.shape)
        self.current_positions += noise
        # move back to center
        self.target_positions -= np.mean(self.target_positions, axis=0)
        self.current_positions -= np.mean(self.current_positions, axis=0)
        self.steps = 0

        obs = self._get_obs()
        obs = np.array(obs, dtype=np.float32).flatten()
        info = {}

        return obs, info

    def _get_obs(self):
        """Flatten positions into 1D observation vector."""
        obs = np.concatenate([
            self.current_positions.flatten(),
            self.target_positions.flatten()
        ])

        return obs.astype(np.float32)

    def step(self, action):
        self.steps += 1
        a_y, a_z = action
        R = action_to_matrix(a_y, a_z, self.max_angle)
        self.current_positions = self.current_positions @ R.T
        self.current_positions -= np.mean(self.current_positions, axis=0)

        # Compute reward: negative average distance between current and target
        dists = np.linalg.norm(self.current_positions - self.target_positions, axis=1)
        #reward = -np.mean(dists)

        R = kabsch_rotation(self.current_positions, self.target_positions)
        angle = np.arccos((np.trace(R) - 1) / 2)

        reward = -angle

        terminated = np.mean(dists) < 0.01
        #terminated = False # prevent early stopping
        truncated = self.steps >= self.max_steps
        info = {"mean_distance": np.mean(dists)}

        obs = self._get_obs()
        obs = np.array(obs, dtype=np.float32).flatten()

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass  # optional — can visualize body positions in matplotlib later
