from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import SAC

from rigid_body_perturbed_env_par import RigidBodyRotationEnv
from stable_baselines3.common.type_aliases import ReplayBufferSamples

import os
import numpy as np
import torch as th
import time


# ============================================================
# SO(2) ROTATION HELPERS
# ============================================================

def rotate_batch_obs(obs_batch, phi_batch, N_atoms):
    """
    obs_batch:
        shape (B, obs_dim)

    phi_batch:
        shape (B,)
    """

    B = obs_batch.shape[0]

    pos_dim = 3 * N_atoms

    current = obs_batch[:, :pos_dim]
    target = obs_batch[:, pos_dim:]

    current = current.reshape(B, N_atoms, 3).copy()
    target = target.reshape(B, N_atoms, 3).copy()

    c = np.cos(phi_batch)[:, None]
    s = np.sin(phi_batch)[:, None]

    # ---- current ----

    y = current[:, :, 1].copy()
    z = current[:, :, 2].copy()

    current[:, :, 1] = c * y - s * z
    current[:, :, 2] = s * y + c * z

    # ---- target ----

    y = target[:, :, 1].copy()
    z = target[:, :, 2].copy()

    target[:, :, 1] = c * y - s * z
    target[:, :, 2] = s * y + c * z

    current = current.reshape(B, pos_dim)
    target = target.reshape(B, pos_dim)

    return np.concatenate([current, target], axis=1)


def rotate_batch_actions(actions, phi_batch):
    """
    actions:
        shape (B,2)
    """

    c = np.cos(phi_batch)
    s = np.sin(phi_batch)

    pxy = actions[:, 0]
    pxz = actions[:, 1]

    out = np.empty_like(actions)

    out[:, 0] = c * pxy - s * pxz
    out[:, 1] = s * pxy + c * pxz

    return out.astype(np.float32)


# ============================================================
# SO(2) AUGMENTED REPLAY BUFFER
# ============================================================

class SO2ReplayBuffer(ReplayBuffer):

    def __init__(self, *args, N_atoms=166, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_atoms = N_atoms

    def sample(self, batch_size, env=None):

        # -------------------------------------------------
        # original batch
        # -------------------------------------------------
        batch = super().sample(batch_size, env)

        obs = batch.observations.cpu().numpy()
        next_obs = batch.next_observations.cpu().numpy()
        actions = batch.actions.cpu().numpy()

        # -------------------------------------------------
        # random SO(2) rotations
        # -------------------------------------------------
        phi = np.random.uniform(
            0,
            2 * np.pi,
            size=batch_size
        )

        obs_rot = rotate_batch_obs(obs, phi, self.N_atoms)
        next_obs_rot = rotate_batch_obs(next_obs, phi, self.N_atoms)
        actions_rot = rotate_batch_actions(actions, phi)

        # -------------------------------------------------
        # numpy -> torch
        # -------------------------------------------------
        device = batch.observations.device

        observations = th.as_tensor(
            obs_rot,
            device=device,
            dtype=batch.observations.dtype
        )

        next_observations = th.as_tensor(
            next_obs_rot,
            device=device,
            dtype=batch.next_observations.dtype
        )

        actions = th.as_tensor(
            actions_rot,
            device=device,
            dtype=batch.actions.dtype
        )

        # rewards and dones unchanged
        rewards = batch.rewards
        dones = batch.dones

        return ReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards
        )

# ============================================================
# CALLBACKS
# ============================================================

def make_env(env_id, N):

    def _init():
        env = RigidBodyRotationEnv(
            env_id=env_id,
            N=N
        )
        return Monitor(env)

    return _init


class PeriodicSaveCallback(BaseCallback):

    def __init__(
        self,
        save_freqs,
        save_path,
        time0,
        timestep_file,
        verbose=1
    ):
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

                model_path = f"{self.save_path}/model_{f_step}"

                self.model.save(model_path)

                #self.model.save_replay_buffer(
                #    f"{self.save_path}/replay_buffer_{f_step}.pkl"
                #)

                self.saved.add(f_step)

                if self.verbose:
                    print(f"Saved model + replay buffer at step {t}")

        return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    N = 166
    num_envs = 4

    env = SubprocVecEnv([
        make_env(i, N)
        for i in range(num_envs)
    ])

    eval_env = SubprocVecEnv([
        make_env(1000 + i, N)
        for i in range(num_envs)
    ])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs_SAC_long/",
        log_path="./logs_SAC_perturbed/",
        eval_freq=200,
        deterministic=True,
        render=False
    )

    time0 = time.time()

    save_list = [
        100,
        1_000,
        2_000,
        5_000,
        10_000,
        #15_000,
        20_000,
        40_000,
        60_000,
        100_000,
        200_000,
        300_000,
        400_000,
        500_000
    ]

    save_callback = PeriodicSaveCallback(
        save_freqs=save_list,
        save_path="./checkpoints_SAC_perturbed/",
        time0=time0,
        timestep_file="timestep.txt",
    )

    policy_kwargs = dict(
        net_arch=[512, 512]
    )

    model = SAC(
        "MlpPolicy",
        env,
        replay_buffer_class=SO2ReplayBuffer,
        replay_buffer_kwargs=dict(
            N_atoms=N
        ),
        verbose=1,
        gradient_steps=4,
        batch_size=512,
        learning_rate=3e-4,
        buffer_size=1000000,
        policy_kwargs=policy_kwargs
    )

    model.learn(
        total_timesteps=500_000,
        callback=[eval_callback, save_callback]
    )

    model.save("models/sac_long300_perturbed005")