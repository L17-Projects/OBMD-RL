import os
import re
import glob
import numpy as np
import pandas as pd

from stable_baselines3 import SAC
from rigid_body_perturbed_env import RigidBodyRotationEnv
from SACpar import SO2ReplayBuffer


import warnings
warnings.filterwarnings("ignore")


# ============================================================
# GEOMETRY UTIL
# ============================================================

def kabsch_rotation(P, Q):
    """
    Optimal rotation aligning P -> Q.
    """
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def rotation_angle_from_matrix(R):
    """
    Returns angle between two configurations.
    """
    trace = np.clip(np.trace(R), -1.0, 3.0)
    return np.arccos((trace - 1) / 2)


# ============================================================
# SINGLE EPISODE EVALUATION
# ============================================================



def run_single_test(model, steps=30, N=4, actions={}):
    env = RigidBodyRotationEnv(N=N)
    obs, _ = env.reset()

    target = env.target_positions

    # initial alignment
    R = kabsch_rotation(env.current_positions, target)
    init_angle = rotation_angle_from_matrix(R)
    # actions as a dictionary. keys are steps, values are the actions taken at that step (lists)
    #actions = {}
    #actions = {} #np.zeros((steps, 2))  # assuming action space of size 2
    # rollout
    for st in range(steps):
        #print(f'Step {st} for model {model} ...') 
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if st not in actions:
            actions[st] = []
        actions[st].append(action)
        if terminated or truncated:
            break

    # final alignment
    R = kabsch_rotation(env.current_positions, target)
    final_angle = rotation_angle_from_matrix(R)

    env.close()
    return init_angle, final_angle, actions


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model_path, n_tests=100, steps=30, N=166, so2=False):
    if(so2):
        print("Loading model with SO(2) replay buffer...")
        model = SAC.load(model_path, device="cpu", custom_objects={
            "replay_buffer_class": SO2ReplayBuffer,
            "replay_buffer_kwargs": {"N_atoms": 166}
        })
    else:
        print("Loading model with standard replay buffer...")
        model = SAC.load(model_path, device="cpu")

    results = np.zeros((n_tests, 2))
    actions = {}
    #all_actions = {}
    # sample the actions as well
    #actions = np.zeros((n_tests, steps, 2))  # assuming action space of size 2
    for i in range(n_tests):
        init_angle, final_angle, actions = run_single_test(model, steps, N, actions)
        results[i, 0] = init_angle
        results[i, 1] = final_angle
        #all_actions[i] = actions


    return results, actions


def summarize(results):
    df = pd.DataFrame(results, columns=["init_angle", "final_angle"])

    return {
        "mean_init": df["init_angle"].mean(),
        "mean_final": df["final_angle"].mean(),
        "improvement": df["final_angle"].mean() - df["init_angle"].mean(),
        "std_final": df["final_angle"].std()
    }


# ============================================================
# CHECKPOINT HANDLING
# ============================================================

def extract_step(path):
    name = os.path.basename(path)
    match = re.search(r"model_(\d+)\.zip", name)
    return int(match.group(1)) if match else -1


def get_checkpoints(folder):
    files = glob.glob(os.path.join(folder, "model_*.zip"))
    return sorted(files, key=extract_step)


# ============================================================
# FULL BENCHMARK
# ============================================================
import datetime as date

def evaluate_all_checkpoints(
    folder="./checkpoints_DDPG_perturbed",
    n_tests=100,
    steps=30,
    N=166,
    save_csv=True,
    res_file='all_results.npy',
    so2=False,
    out_file="checkpoint_results.csv"
):

    checkpoints = get_checkpoints(folder)

    all_stats = []

    print(f"Found {len(checkpoints)} checkpoints\n")
    all_results = []

    all_actions = []

    # start time
    
    for path in checkpoints:
        start_time = date.datetime.now()
        step = extract_step(path)
        print(f"Evaluating step {step} ...")

        results, actions = evaluate_model(path, n_tests=n_tests, steps=steps, N=N, so2=so2)
        stats = summarize(results)

        stats.update({
            "step": step,
            "model_path": path
        })
        all_results.append(results)

        all_stats.append(stats)

        # save actions dictionary for this checkpoint
        actions_file = f"actions_step_{step}.npy"
        np.save(actions_file, actions)
        end_time = date.datetime.now()
        elapsed_time = end_time - start_time
        print(
            f"step {step}: "
            f"improvement={stats['improvement']:.4f}, "
            f"final={stats['mean_final']:.4f}"
        )

        print(f"Elapsed time for step {step}: {elapsed_time}\n")

    df = pd.DataFrame(all_stats)
    df = df.sort_values("step")

    if save_csv:
        df.to_csv(out_file, index=False)
        print(f"\nSaved results to {out_file}")

    np.save(res_file, np.array(all_results))
    return df, all_results, all_actions


if __name__ == "__main__":  

    CHECKPOINT_DIR = "./checkpoints_SAC_perturbed"
    df, all_results, actions = evaluate_all_checkpoints(
        folder=CHECKPOINT_DIR,
        n_tests=100,   # increase for final reporting
        steps=30,
        N=166,
        save_csv=True,
        res_file='all_results.npy',
        so2=True,
        out_file="checkpoint_results.csv"
    )

