import time
import numpy as np
import torch as th

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from protein_env_prll import ProteinEnv


# ============================================================
# ENV FACTORY
# ============================================================

def make_env(env_id):
    def _init():
        env = ProteinEnv(
            env_id=env_id,
            template_data="lammps/ubq_start.data",
            lammps_exe=None,
            nprocs=1
        )
        return Monitor(env)
    return _init


# ============================================================
# SO(2) ROTATIONS (NO N_atoms REQUIRED)
# ============================================================

def rotate_xyz(flat_xyz, phi):
    """
    flat_xyz: (B, 3N)
    """

    B, dim = flat_xyz.shape
    N = dim // 3

    xyz = flat_xyz.reshape(B, N, 3).copy()

    c = np.cos(phi)[:, None]
    s = np.sin(phi)[:, None]

    y = xyz[:, :, 1].copy()
    z = xyz[:, :, 2].copy()

    xyz[:, :, 1] = c * y - s * z
    xyz[:, :, 2] = s * y + c * z

    return xyz.reshape(B, dim)


def rotate_obs(obs, phi):
    """
    obs: (B, 6N) = [current, target]
    """

    dim = obs.shape[1]
    half = dim // 2

    current = obs[:, :half]
    target = obs[:, half:]

    current_rot = rotate_xyz(current, phi)
    target_rot = rotate_xyz(target, phi)

    return np.concatenate([current_rot, target_rot], axis=1)


def rotate_actions(actions, phi):
    """
    actions: (B, 2) = [pxy, pxz]
    """

    c = np.cos(phi)
    s = np.sin(phi)

    out = np.empty_like(actions)

    pxy = actions[:, 0]
    pxz = actions[:, 1]

    out[:, 0] = c * pxy - s * pxz
    out[:, 1] = s * pxy + c * pxz

    return out.astype(np.float32)


# ============================================================
# REPLAY BUFFER WITH SYMMETRY AUGMENTATION
# ============================================================

class ProteinSO2ReplayBuffer(ReplayBuffer):

    def sample(self, batch_size, env=None):

        batch = super().sample(batch_size, env)

        obs = batch.observations.cpu().numpy()
        next_obs = batch.next_observations.cpu().numpy()
        actions = batch.actions.cpu().numpy()

        phi = np.random.uniform(0.0, 2.0 * np.pi, size=batch_size)

        obs_rot = rotate_obs(obs, phi)
        next_obs_rot = rotate_obs(next_obs, phi)
        actions_rot = rotate_actions(actions, phi)

        device = batch.observations.device

        return ReplayBufferSamples(
            observations=th.as_tensor(obs_rot, device=device, dtype=batch.observations.dtype),
            actions=th.as_tensor(actions_rot, device=device, dtype=batch.actions.dtype),
            next_observations=th.as_tensor(next_obs_rot, device=device, dtype=batch.next_observations.dtype),
            rewards=batch.rewards,
            dones=batch.dones
        )


# ============================================================
# CALLBACK
# ============================================================

class PeriodicSaveCallback(BaseCallback):

    def __init__(self, save_freqs, save_path, time0, timestep_file, verbose=1):
        super().__init__(verbose)
        self.save_freqs = sorted(save_freqs)
        self.save_path = save_path
        self.timestep_file = timestep_file
        self.saved = set()
        self.time0 = time0

    def _on_step(self) -> bool:

        t = self.model.num_timesteps

        with open(self.timestep_file, "a") as f:
            f.write(f"{t} {time.time() - self.time0:.2f}\n")

        for f_step in self.save_freqs:
            if t >= f_step and f_step not in self.saved:

                self.model.save(f"{self.save_path}/model_{f_step}")

                self.saved.add(f_step)

                if self.verbose:
                    print(f"Saved model at step {t}")

        return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    num_envs = 32

    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    eval_env = SubprocVecEnv([make_env(1000 + i) for i in range(num_envs)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs_SAC_protein/",
        log_path="./logs_SAC_protein/",
        eval_freq=200,
        deterministic=True,
        render=False
    )

    time0 = time.time()

    save_list = [
        1_000, 5_000, 10_000, 15_000,
        20_000, 40_000, 60_000,
        100_000, 200_000, 400_000
    ]

    save_callback = PeriodicSaveCallback(
        save_freqs=save_list,
        save_path="./checkpoints_SAC/",
        time0=time0,
        timestep_file="timestep.txt",
    )

    policy_kwargs = dict(net_arch=[512, 512])

    model = SAC(
        "MlpPolicy",
        env,
        replay_buffer_class=ProteinSO2ReplayBuffer,
        buffer_size=1_000_000,
        batch_size=512,
        learning_rate=3e-4,
        gradient_steps=32,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    model.learn(
        total_timesteps=500_000,
        callback=[eval_callback, save_callback]
    )

    model.save("models/sac_protein_so2")

    print("\nTraining finished.\n")