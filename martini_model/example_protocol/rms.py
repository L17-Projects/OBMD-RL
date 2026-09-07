#!/usr/bin/env python3

from pathlib import Path
import csv
import math
import re

import numpy as np


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path("lammps")
MODEL_DIR = BASE_DIR / "model_400000"

TRAJECTORY_NAME = "protein.lammpstrj"

COMBINED_OUTPUT = Path("./")/ "rmsd_all.csv"


# ============================================================
# KABSCH
# ============================================================

def kabsch_rotation(current_positions, target_positions):
    """
    Computes the optimal rotation matrix that aligns
    current_positions to target_positions using the Kabsch algorithm.
    """

    H = current_positions.T @ target_positions

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


# ============================================================
# READ LAMMPS TRAJECTORY
# ============================================================

def read_trajectory(filename):
    """
    Read a LAMMPS trajectory.

    Expected atom columns:
        id type x y z

    Atoms are sorted by ID in every frame.
    """

    frames = []

    with open(filename, "r") as f:

        while True:

            line = f.readline()

            if not line:
                break

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(f.readline().strip())

            # Number of atoms
            line = f.readline()

            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError(
                    f"Unexpected format at timestep {timestep}"
                )

            n_atoms = int(f.readline().strip())

            # Box bounds
            line = f.readline()

            if not line.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError(
                    f"Expected BOX BOUNDS at timestep {timestep}"
                )

            f.readline()
            f.readline()
            f.readline()

            # Atom header
            line = f.readline()

            if not line.startswith("ITEM: ATOMS"):
                raise RuntimeError(
                    f"Expected ATOMS section at timestep {timestep}"
                )

            columns = line.strip().split()[2:]

            try:
                id_index = columns.index("id")
                x_index = columns.index("x")
                y_index = columns.index("y")
                z_index = columns.index("z")
            except ValueError:
                raise RuntimeError(
                    f"Could not find id/x/y/z columns.\n"
                    f"Available columns: {columns}"
                )

            # Read atoms
            atom_data = []

            for _ in range(n_atoms):

                fields = f.readline().split()

                atom_id = int(fields[id_index])

                x = float(fields[x_index])
                y = float(fields[y_index])
                z = float(fields[z_index])

                atom_data.append(
                    (atom_id, x, y, z)
                )

            # Sort by atom ID
            atom_data.sort(
                key=lambda x: x[0]
            )

            atom_ids = [
                atom[0]
                for atom in atom_data
            ]

            # Check IDs are unique
            if len(set(atom_ids)) != n_atoms:
                raise RuntimeError(
                    f"Duplicate atom IDs at timestep "
                    f"{timestep}"
                )

            coordinates = np.array(
                [
                    [atom[1], atom[2], atom[3]]
                    for atom in atom_data
                ],
                dtype=float
            )

            frames.append({
                "timestep": timestep,
                "atom_ids": atom_ids,
                "coordinates": coordinates
            })

    return frames


# ============================================================
# RMSD
# ============================================================

def calculate_rmsd(current, reference):
    """
    Calculate RMSD between two centered and aligned
    coordinate sets.
    """

    difference = current - reference

    squared_distances = np.sum(
        difference ** 2,
        axis=1
    )

    return np.sqrt(
        np.mean(squared_distances)
    )


# ============================================================
# PROCESS ONE TRAJECTORY
# ============================================================

def process_trajectory(filename):
    """
    Calculate RMSD relative to the initial protein structure
    for every frame.

    The structures are centered and optimally aligned using
    the Kabsch algorithm before calculating RMSD.
    """

    frames = read_trajectory(filename)

    if not frames:
        raise RuntimeError(
            f"No frames found in {filename}"
        )

    # --------------------------------------------------------
    # Reference structure = first frame
    # --------------------------------------------------------

    reference = frames[0]["coordinates"]

    reference_ids = frames[0]["atom_ids"]

    # Center reference
    reference_centered = (
        reference
        - reference.mean(axis=0)
    )

    results = []

    # --------------------------------------------------------
    # Process every frame
    # --------------------------------------------------------

    for frame_number, frame in enumerate(frames):

        current = frame["coordinates"]
        current_ids = frame["atom_ids"]

        timestep = frame["timestep"]

        # Check atom count
        if current.shape != reference.shape:
            raise RuntimeError(
                f"Atom count changed at timestep "
                f"{timestep}"
            )

        # Check atom IDs
        if current_ids != reference_ids:
            raise RuntimeError(
                f"Atom IDs/order changed at timestep "
                f"{timestep}"
            )

        # ----------------------------------------------------
        # Center current structure
        # ----------------------------------------------------

        current_centered = (
            current
            - current.mean(axis=0)
        )

        # ----------------------------------------------------
        # Kabsch alignment
        #
        # current -> reference
        # ----------------------------------------------------

        R = kabsch_rotation(
            current_centered,
            reference_centered
        )

        # Apply rotation
        aligned = (
            current_centered @ R.T
        )

        # ----------------------------------------------------
        # RMSD
        # ----------------------------------------------------

        rmsd = calculate_rmsd(
            aligned,
            reference_centered
        )

        results.append({
            "frame": frame_number,
            "timestep": timestep,
            "rmsd": rmsd
        })

    return results


# ============================================================
# FIND EVAL DIRECTORIES
# ============================================================

def find_eval_directories():

    eval_dirs = []

    for path in MODEL_DIR.iterdir():

        if not path.is_dir():
            continue

        if re.fullmatch(r"eval_\d+", path.name):
            eval_dirs.append(path)

    eval_dirs.sort(
        key=lambda p: int(
            p.name.split("_")[1]
        )
    )

    return eval_dirs


# ============================================================
# MAIN
# ============================================================

def main():

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}"
        )

    eval_dirs = find_eval_directories()

    if not eval_dirs:
        raise RuntimeError(
            f"No eval_* directories found in {MODEL_DIR}"
        )

    all_results = []

    for eval_dir in eval_dirs:

        trajectory = (
            eval_dir / TRAJECTORY_NAME
        )

        if not trajectory.exists():

            print(
                f"WARNING: {trajectory} does not exist "
                "-- skipping."
            )

            continue

        print(
            f"Processing {eval_dir.name} ..."
        )

        results = process_trajectory(
            trajectory
        )

        # Per-evaluation output
        output_file = (
            eval_dir / "rmsd.csv"
        )

        with open(
            output_file,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "frame",
                "timestep",
                "rmsd"
            ])

            for r in results:

                writer.writerow([
                    r["frame"],
                    r["timestep"],
                    f"{r['rmsd']:.10f}"
                ])

        # Combined results
        for r in results:

            all_results.append([
                eval_dir.name,
                r["frame"],
                r["timestep"],
                r["rmsd"]
            ])

    # --------------------------------------------------------
    # Sort by eval, then frame
    # --------------------------------------------------------

    all_results.sort(
        key=lambda x: (
            int(x[0].split("_")[1]),
            int(x[1])
        )
    )

    # --------------------------------------------------------
    # Combined CSV
    # --------------------------------------------------------

    with open(
        COMBINED_OUTPUT,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "eval",
            "frame",
            "timestep",
            "rmsd"
        ])

        for row in all_results:

            writer.writerow([
                row[0],
                row[1],
                row[2],
                f"{row[3]:.10f}"
            ])

    print()
    print("Done.")
    print(
        f"Combined output: {COMBINED_OUTPUT}"
    )


if __name__ == "__main__":
    main()