from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from protein_env_prll_perf import ProteinEnv
from stable_baselines3 import SAC, TD3
import numpy as np
import time
import sys
import os
import shutil
from collections import defaultdict
#from SAC_prll import ProteinSO2ReplayBuffer


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

import numpy as np 
import matplotlib.pyplot as plt


def data_to_xyz(data_file, end_file, ind):
    with open(data_file, 'r') as f:
        lines = f.readlines()

    # Find the line where the atom coordinates start
    start_index = None
    for i, line in enumerate(lines):
        if "Atoms" in line.split():
            start_index = i + 2  # Skip the "Atoms" line and the blank line after it
            break

    if start_index is None:
        raise ValueError("Could not find 'Atoms' section in the data file.")

    # Extract atom coordinates
    positions = []

    # index as string with 3 leeding zeros
    ind_str = str(ind).zfill(3)
    
    xyz_file = f'{end_file}_{ind_str}.xyz'

    for line in lines[start_index:]:
        if line.strip() == "":
            break  # Stop at the first blank line after the coordinates
        parts = line.split()
        if(int(parts[2]) < 19):
            idd, typee, x, y, z = int(parts[0]), int(parts[2]), float(parts[4]), float(parts[5]), float(parts[6])  # Assuming format: id type x y z
            positions.append([idd, typee, x, y, z])
    # sort positions by type

    # calculate cms of positions and subtract from positions
    positions = np.array(positions)
    cms = np.mean(positions[:, 2:], axis=0)
    positions[:, 2:] -= cms
    positions = positions[positions[:, 0].argsort()]
    with open(xyz_file, 'w') as f:
        f.write(f"{len(positions)}\n")
        f.write("# start configuration\n")
        for i, pos in enumerate(positions):
            f.write(f"{pos[1]} {pos[2]} {pos[3]} {pos[4]}\n")  # Assuming all atoms are carbon for XYZ format
    return np.array([pos[1] for pos in positions])

def target_to_xyz(target_positions, end_file, ind, id_list):
    ind_str = str(ind).zfill(3)
    xyz_file = f'{end_file}_{ind_str}.xyz'
    with open(xyz_file, 'w') as f:
        f.write(f"{len(target_positions)}\n")
        f.write("# target configuration\n")
        for i, pos in zip(id_list, target_positions):
            f.write(f"{i} {pos[0]} {pos[1]} {pos[2]}\n")  # Assuming all atoms are carbon for XYZ format


def evaluate_model(model_name, model_type, N_episodes=200, start_index=0, workdir="lammps"):
    if model_type == "SAC":
        model = SAC.load(f"models/{model_name}")
    elif model_type == "TD3":
        model = TD3.load(f"models/{model_name}")
    else:
        raise ValueError("Invalid model type. Choose 'SAC' or 'TD3'.")
        
    # Wrap environment in Monitor to record episode stats
    env = ProteinEnv(workdir=workdir)
    action_dict = defaultdict(list)
    print("model loaded, entering evaluation loop")

    average_initial_dot_products = []
    average_final_dot_products = []
    angles_over_episodes = []
    for k in range(N_episodes):
        obs, _ = env.reset()
        product_values = []
        # target postions
        obs_target = obs[obs.shape[0]//2:].reshape(-1, 3)

        target_unnorm = env._get_unnormalized_target()

        # kopira začetno stanje
        #shutil.copy(f"{workdir}/rotated.data", f"{workdir}/start{k+1}.data") 
        wd = os.path.abspath(workdir)
        #shutil.copy(f"{wd}/../dfr_start.data", f"{wd}/dfr_start.data") # mod 16.4.2026
        # moras jih sortirati prej: TODO !!!
        id_list = data_to_xyz(f"{wd}/rotated.data", f"{wd}/start", k+1)
        target_to_xyz(target_unnorm, f"{wd}/target", k+1, id_list)
        # save dot product before any steps
        obs_current = obs[:obs.shape[0]//2].reshape(-1, 3)

        dot_products = []
        #for i in range(obs_current.shape[0]):
        #    vec_current_norm = obs_current[i] / np.linalg.norm(obs_current[i])
        #    vec_target_norm = obs_target[i] / np.linalg.norm(obs_target[i])
        #    dot_products.append(np.dot(vec_current_norm, vec_target_norm))

        R = kabsch_rotation(obs_current, obs_target)
        angle = np.arccos((np.trace(R) - 1) / 2)
        print("Initial angle:", angle)

        product_values.append(angle)

        for j in range(30): # modified 9.3.2026: 15->30 steps per episode. 9.3. 15 je dovolj. za vsak slucaj 20

            #print(f'CWD = {os.getcwd()}')
            #time.sleep(2)
            

            action, _states = model.predict(obs, deterministic=True)
            action_dict[j].append(action)

            print('predicted actions are:', action)

            obs, reward, terminated, truncated, info = env.step(action*(-1)) #mod. 8.3. ... pomoje ker je bil prej testiran model z rotacijskimi matrikami

            obs_current = obs[:obs.shape[0]//2].reshape(-1, 3)

            # print average dot procuct of each vector from current to target
            # dot_products = []
            # for i in range(obs_current.shape[0]):
            #    vec_current_norm = obs_current[i] / np.linalg.norm(obs_current[i])
            #    vec_target_norm = obs_target[i] / np.linalg.norm(obs_target[i])
            #    dot_products.append(np.dot(vec_current_norm, vec_target_norm))

            R = kabsch_rotation(obs_current, obs_target)
            angle = np.arccos((np.trace(R) - 1) / 2)
            print("Angle after step:", angle)

            product_values.append(angle)

            if terminated or truncated:
                obs, _ = env.reset()
                print("Episode finished.")
        print("Angles over episode:", product_values)
        angles_over_episodes.append(product_values)
        average_initial_dot_products.append(product_values[0])
        average_final_dot_products.append(product_values[-1])

        
        #data_to_xyz(f"{workdir}/rotated.data", f"{workdir}/", k+1)
        #shutil.copy(f"{workdir}/rotated.data", f"{workdir}/target{k+1}.data") 

    # save initial and final average dot products to npy file
    average_initial_dot_products = np.array(average_initial_dot_products)
    average_final_dot_products = np.array(average_final_dot_products)
    np.save(f"models/results_initial_{model_name}_{start_index}.npy", average_initial_dot_products)
    np.save(f"models/results_final_{model_name}_{start_index}.npy", average_final_dot_products)
    np.save(f"models/angles_over_episodes_{model_name}_{start_index}.npy", np.array(angles_over_episodes))
    np.save(f"models/action_dict_{model_name}_{start_index}.npy",dict(action_dict),allow_pickle=True)


if __name__ == "__main__":
    model_name = sys.argv[1]
    model_type = sys.argv[2]
    start_ep = int(sys.argv[3])
    end_ep = int(sys.argv[4])
    workdir = sys.argv[5]

    evaluate_model(model_name, model_type, N_episodes=(end_ep - start_ep), start_index=start_ep, workdir=workdir)
