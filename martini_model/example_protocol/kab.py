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

COMBINED_OUTPUT = MODEL_DIR / "kabsch_angles_all.csv"

# Target rotation
TARGET_ANGLE = np.pi / 2.0

# Target rotation axis
TARGET_AXIS = np.array([1.0, 1.0, 1.0])
TARGET_AXIS /= np.linalg.norm(TARGET_AXIS)


# ============================================================
# READ LAMMPS TRAJECTORY
# ============================================================

def read_trajectory(filename):
    """
    Read a LAMMPS trajectory and sort atoms by ID in every frame.

    Expected atom columns:
        id type x y z

    Returns a list of dictionaries containing:
        timestep
        coordinates
    """

    frames = []

    with open(filename, "r") as f:

        while True:

            line = f.readline()

            if not line:
                break

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            # ------------------------------------------------
            # Timestep
            # ------------------------------------------------

            timestep = int(f.readline().strip())

            # ------------------------------------------------
            # Number of atoms
            # ------------------------------------------------

            line = f.readline()

            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError(
                    f"Unexpected format at timestep {timestep}"
                )

            n_atoms = int(f.readline().strip())

            # ------------------------------------------------
            # Box bounds
            # ------------------------------------------------

            line = f.readline()

            if not line.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError(
                    f"Expected BOX BOUNDS at timestep {timestep}"
                )

            # Read the three box-bound lines
            f.readline()
            f.readline()
            f.readline()

            # ------------------------------------------------
            # Atom header
            # ------------------------------------------------

            line = f.readline()

            if not line.startswith("ITEM: ATOMS"):
                raise RuntimeError(
                    f"Expected ATOMS section at timestep {timestep}"
                )

            columns = line.strip().split()[2:]

            # Find required columns
            try:
                id_index = columns.index("id")
                x_index = columns.index("x")
                y_index = columns.index("y")
                z_index = columns.index("z")
            except ValueError as exc:
                raise RuntimeError(
                    f"Could not find required columns at timestep "
                    f"{timestep}.\n"
                    f"Expected: id type x y z\n"
                    f"Available columns: {columns}"
                ) from exc

            # ------------------------------------------------
            # Read atoms
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Sort atoms by ID
            # ------------------------------------------------

            atom_data.sort(
                key=lambda atom: atom[0]
            )

            # ------------------------------------------------
            # Extract sorted coordinates
            # ------------------------------------------------

            coords = np.array(
                [
                    [x, y, z]
                    for atom_id, x, y, z in atom_data
                ],
                dtype=float
            )

            # ------------------------------------------------
            # Check that IDs are unique
            # ------------------------------------------------

            atom_ids = [
                atom_id
                for atom_id, x, y, z in atom_data
            ]

            if len(set(atom_ids)) != n_atoms:
                raise RuntimeError(
                    f"Duplicate atom IDs found at timestep "
                    f"{timestep}"
                )

            frames.append({
                "timestep": timestep,
                "coordinates": coords
            })

    return frames


# ============================================================
# ROTATION MATRIX
# ============================================================

def axis_angle_rotation_matrix(axis, angle):
    """
    Rodrigues rotation formula.

    Returns the 3x3 rotation matrix corresponding to
    rotation by 'angle' around 'axis'.
    """

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis

    K = np.array([
        [0.0, -z,  y],
        [z,   0.0, -x],
        [-y,  x,   0.0]
    ])

    I = np.eye(3)

    R = (
        I
        + np.sin(angle) * K
        + (1.0 - np.cos(angle)) * (K @ K)
    )

    return R


# ============================================================
# CONSTRUCT TARGET
# ============================================================

def make_target(initial_coordinates):
    """
    Construct the target configuration.

    The target is the initial configuration rotated by
    pi/2 around the axis (1,1,1).

    The structure is first centered so that the rotation
    is performed around its center of geometry.
    """

    # Center initial structure
    centered = (
        initial_coordinates
        - initial_coordinates.mean(axis=0)
    )

    # Rotation matrix
    R_target = axis_angle_rotation_matrix(
        TARGET_AXIS,
        TARGET_ANGLE
    )

    # Rotate
    target = centered @ R_target.T

    return target


# ============================================================
# KABSCH ROTATION
# ============================================================

def kabsch_rotation(P, Q):
    """
    Calculate the optimal rotation mapping P onto Q.

    P and Q must be centered N x 3 coordinate arrays.

    Returns
    -------
    R : 3x3 numpy array
        Optimal proper rotation matrix.
    """

    H = P.T @ Q

    U, S, Vt = np.linalg.svd(H)

    # Ensure a proper rotation rather than reflection
    D = np.eye(3)

    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1

    R = Vt.T @ D @ U.T

    return R


# ============================================================
# ROTATION ANGLE
# ============================================================

def rotation_matrix_to_angle(R):
    """
    Extract the rotation angle from a rotation matrix.

    Returns angle in radians in [0, pi].
    """

    cos_angle = (
        np.trace(R) - 1.0
    ) / 2.0

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0
    )

    return math.acos(cos_angle)


# ============================================================
# PROCESS ONE TRAJECTORY
# ============================================================

def process_trajectory(filename):
    """
    Calculate the Kabsch angle between every trajectory frame
    and the target configuration.
    """

    frames = read_trajectory(filename)

    if len(frames) == 0:
        raise RuntimeError(
            f"No frames found in {filename}"
        )

    # --------------------------------------------------------
    # Initial configuration
    # --------------------------------------------------------

    initial = frames[0]["coordinates"]

    # --------------------------------------------------------
    # Construct target
    # --------------------------------------------------------

    target = make_target(initial)

    results = []

    # --------------------------------------------------------
    # Compare every frame to target
    # --------------------------------------------------------

    for frame_number, frame in enumerate(frames):

        timestep = frame["timestep"]

        current = frame["coordinates"]

        if current.shape != target.shape:
            raise RuntimeError(
                f"Atom count changed at timestep {timestep}"
            )

        # Center current structure
        current_centered = (
            current
            - current.mean(axis=0)
        )

        # Kabsch rotation:
        #
        # current -> target
        #
        R = kabsch_rotation(
            current_centered,
            target
        )

        # Angle required to optimally rotate
        # current into target
        angle = rotation_matrix_to_angle(R)

        results.append({
            "frame": frame_number,
            "timestep": timestep,
            "kabsch_angle_rad": angle,
            "kabsch_angle_deg": math.degrees(angle)
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

    print(
        f"Found {len(eval_dirs)} evaluation directories."
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

        # ----------------------------------------------------
        # Per-evaluation CSV
        # ----------------------------------------------------

        output_file = (
            eval_dir / "kabsch_angle.csv"
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
                "kabsch_angle_rad",
                "kabsch_angle_deg"
            ])

            for r in results:

                writer.writerow([
                    r["frame"],
                    r["timestep"],
                    f"{r['kabsch_angle_rad']:.10f}",
                    f"{r['kabsch_angle_deg']:.6f}"
                ])

        print(
            f"  {len(results)} frames -> "
            f"{output_file}"
        )

        # ----------------------------------------------------
        # Combined results
        # ----------------------------------------------------

        for r in results:

            all_results.append([
                eval_dir.name,
                r["frame"],
                r["timestep"],
                r["kabsch_angle_rad"],
                r["kabsch_angle_deg"]
            ])

    # --------------------------------------------------------
    # Write combined CSV
    # --------------------------------------------------------

    all_results.sort(
        key=lambda x: (
            int(x[0].split("_")[1]),  # eval number
            x[1]                      # frame number
        )
    )
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
            "kabsch_angle_rad",
            "kabsch_angle_deg"
        ])

        for row in all_results:

            writer.writerow([
                row[0],
                row[1],
                row[2],
                f"{row[3]:.10f}",
                f"{row[4]:.6f}"
            ])

    print()
    print("Done.")
    print(
        f"Combined output: {COMBINED_OUTPUT}"
    )


if __name__ == "__main__":
    main()