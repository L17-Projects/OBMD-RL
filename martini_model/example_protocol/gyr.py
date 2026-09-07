
#!/usr/bin/env python3

from pathlib import Path
import re
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# Root directory containing model_400000
ROOT = Path("lammps") / "model_400000"

# Name of the trajectory file in each eval_* directory
TRAJ_NAME = "protein.lammpstrj"

# Output directory
OUTPUT_DIR = ROOT / "radius_of_gyration"

# If your trajectory contains multiple molecule/species types and
# you only want part of the system, modify this later.
# Currently, ALL atoms in protein.lammpstrj are used.

# LAMMPS coordinates:
#   "x y z"       -> ordinary (possibly wrapped) coordinates
#   "xu yu zu"    -> unwrapped coordinates
#
# For protein trajectories, unwrapped coordinates are usually preferable
# if the protein can cross periodic boundaries.
USE_UNWRAPPED_IF_AVAILABLE = True


# ============================================================
# LAMMPS TRAJECTORY READER
# ============================================================

def read_lammps_trajectory(filename):
    """
    Generator yielding:
        timestep, atom_ids, coordinates

    for every frame in a LAMMPS dump trajectory.

    Coordinates are returned as an (N, 3) numpy array.
    """

    with open(filename, "r") as f:

        while True:
            line = f.readline()

            if not line:
                break

            # ------------------------------------------------
            # TIMESTEP
            # ------------------------------------------------
            if line.strip() != "ITEM: TIMESTEP":
                continue

            timestep_line = f.readline()
            if not timestep_line:
                break

            timestep = int(timestep_line.strip())

            # ------------------------------------------------
            # NUMBER OF ATOMS
            # ------------------------------------------------
            line = f.readline()
            if line.strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError(
                    f"Unexpected format in {filename}: "
                    f"expected ITEM: NUMBER OF ATOMS"
                )

            n_atoms = int(f.readline().strip())

            # ------------------------------------------------
            # BOX BOUNDS
            # ------------------------------------------------
            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(
                    f"Unexpected format in {filename}: "
                    f"expected ITEM: BOX BOUNDS"
                )

            # Normally 3 lines for x/y/z bounds.
            # We don't actually need the box for Rg if using
            # unwrapped coordinates.
            box_bounds = []
            for _ in range(3):
                box_bounds.append(f.readline().split())

            # ------------------------------------------------
            # ATOMS
            # ------------------------------------------------
            line = f.readline()

            if not line.startswith("ITEM: ATOMS"):
                raise ValueError(
                    f"Unexpected format in {filename}: "
                    f"expected ITEM: ATOMS"
                )

            columns = line.strip().split()[2:]

            # Map column name -> index
            col_index = {name: i for i, name in enumerate(columns)}

            # ------------------------------------------------
            # Find coordinate columns
            # ------------------------------------------------
            if (
                USE_UNWRAPPED_IF_AVAILABLE
                and all(c in col_index for c in ("xu", "yu", "zu"))
            ):
                x_col = col_index["xu"]
                y_col = col_index["yu"]
                z_col = col_index["zu"]

            elif all(c in col_index for c in ("x", "y", "z")):
                x_col = col_index["x"]
                y_col = col_index["y"]
                z_col = col_index["z"]

            else:
                raise ValueError(
                    f"Could not find coordinates in {filename}. "
                    f"Available columns: {columns}"
                )

            # Atom IDs are useful for sorting, especially if the
            # dump does not write atoms in a consistent order.
            id_col = col_index.get("id", None)

            coordinates = np.empty((n_atoms, 3), dtype=float)

            if id_col is not None:
                atom_ids = np.empty(n_atoms, dtype=int)
            else:
                atom_ids = np.arange(n_atoms)

            # ------------------------------------------------
            # Read atoms
            # ------------------------------------------------
            for i in range(n_atoms):
                values = f.readline().split()

                if id_col is not None:
                    atom_ids[i] = int(values[id_col])

                coordinates[i, 0] = float(values[x_col])
                coordinates[i, 1] = float(values[y_col])
                coordinates[i, 2] = float(values[z_col])

            # Sort by atom ID
            if id_col is not None:
                order = np.argsort(atom_ids)
                atom_ids = atom_ids[order]
                coordinates = coordinates[order]

            yield timestep, atom_ids, coordinates


# ============================================================
# RADIUS OF GYRATION
# ============================================================

def radius_of_gyration(coordinates):
    """
    Calculate the radius of gyration for a set of atoms.

    Rg = sqrt(mean(|r_i - COM|^2))

    All atoms are assumed to have equal mass.
    """

    center_of_mass = np.mean(coordinates, axis=0)

    displacement = coordinates - center_of_mass

    rg_squared = np.mean(np.sum(displacement**2, axis=1))

    return np.sqrt(rg_squared)


# ============================================================
# PROCESS ONE TRAJECTORY
# ============================================================

def process_trajectory(filename):
    """
    Calculate Rg for every frame in one trajectory.

    Returns
    -------
    timesteps : numpy array
    rg_values : numpy array
    """

    timesteps = []
    rg_values = []

    print(f"Reading {filename}")

    for frame_number, (timestep, atom_ids, coordinates) in enumerate(
        read_lammps_trajectory(filename), start=1
    ):

        rg = radius_of_gyration(coordinates)

        timesteps.append(timestep)
        rg_values.append(rg)

        if frame_number % 100 == 0:
            print(
                f"    processed {frame_number} frames "
                f"(timestep {timestep})"
            )

    return np.asarray(timesteps), np.asarray(rg_values)


# ============================================================
# MAIN
# ============================================================

def main():

    if not ROOT.exists():
        raise FileNotFoundError(
            f"Could not find directory: {ROOT.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Find eval_0, eval_1, eval_2, ...
    # --------------------------------------------------------
    eval_dirs = []

    for path in ROOT.glob("eval_*"):

        if not path.is_dir():
            continue

        match = re.fullmatch(r"eval_(\d+)", path.name)

        if match:
            eval_number = int(match.group(1))
            eval_dirs.append((eval_number, path))

    eval_dirs.sort()

    if not eval_dirs:
        raise RuntimeError(
            f"No eval_* directories found in {ROOT.resolve()}"
        )

    print(f"Found {len(eval_dirs)} evaluation directories.")

    # --------------------------------------------------------
    # Store everything for combined output
    # --------------------------------------------------------
    all_results = []

    # --------------------------------------------------------
    # Process every eval directory
    # --------------------------------------------------------
    for eval_number, eval_dir in eval_dirs:

        trajectory = eval_dir / TRAJ_NAME

        if not trajectory.exists():
            print(
                f"WARNING: {trajectory} does not exist. Skipping."
            )
            continue

        print()
        print("=" * 60)
        print(f"Processing eval_{eval_number}")
        print("=" * 60)

        timesteps, rg_values = process_trajectory(trajectory)

        # ----------------------------------------------------
        # Write individual CSV
        # ----------------------------------------------------
        output_csv = OUTPUT_DIR / f"eval_{eval_number}_rg.csv"

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "frame",
                "timestep",
                "radius_of_gyration"
            ])

            for frame, (timestep, rg) in enumerate(
                zip(timesteps, rg_values)
            ):
                writer.writerow([
                    frame,
                    timestep,
                    rg
                ])

        print(f"  Frames: {len(rg_values)}")
        print(f"  Rg mean: {np.mean(rg_values):.6f}")
        print(f"  Rg std:  {np.std(rg_values):.6f}")
        print(f"  Saved: {output_csv}")

        # Save for combined CSV
        for frame, (timestep, rg) in enumerate(
            zip(timesteps, rg_values)
        ):
            all_results.append([
                eval_number,
                frame,
                timestep,
                rg
            ])

    # --------------------------------------------------------
    # Combined CSV
    # --------------------------------------------------------
    combined_csv = OUTPUT_DIR / "all_evals_rg.csv"

    with open(combined_csv, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "eval",
            "frame",
            "timestep",
            "radius_of_gyration"
        ])

        writer.writerows(all_results)

    print()
    print(f"Combined results saved to:")
    print(f"  {combined_csv}")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for eval_number, eval_dir in eval_dirs:

        csv_file = OUTPUT_DIR / f"eval_{eval_number}_rg.csv"

        if not csv_file.exists():
            continue

        data = np.loadtxt(
            csv_file,
            delimiter=",",
            skiprows=1
        )

        if data.ndim == 1:
            data = data.reshape(1, -1)

        frame = data[:, 0]
        rg = data[:, 2]

        plt.plot(
            frame,
            rg,
            label=f"eval_{eval_number}",
            alpha=0.7
        )

    plt.xlabel("Frame")
    plt.ylabel(r"Radius of gyration ($R_g$)")
    plt.title("Radius of gyration vs. frame")
    plt.tight_layout()

    plot_file = OUTPUT_DIR / "radius_of_gyration_vs_frame.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()

    print(f"Plot saved to:")
    print(f"  {plot_file}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()


