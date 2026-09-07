#!/usr/bin/env python3

from pathlib import Path
import csv
import os
import re

import numpy as np


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path("lammps")
MODEL_DIR = BASE_DIR / "model_400000"

TRAJECTORY_NAME = "protein.lammpstrj"

COMBINED_OUTPUT = Path("./") / "rmsd_all.csv"


# ------------------------------------------------------------
# REFERENCE LAMMPS DATA FILE
# ------------------------------------------------------------
#
# This file provides the reference coordinates.
#
# Protein atoms have types 1-23.
# Type 24 is water and is excluded.
#

REFERENCE_DATA_FILE = Path("dfr_start.data")

PROTEIN_TYPES = set(range(1, 24))


# ------------------------------------------------------------
# RMSD ATOM IDS
# ------------------------------------------------------------
#
# resid_all.npy contains the atom IDs that should actually
# be used for the RMSD calculation.
#
# Final RMSD atom set:
#
#     IDs in resid_all.npy
#     AND
#     atoms with types 1-23 in dfr_start.data
#
# ------------------------------------------------------------

BACKBONE_IDS_FILE = Path("resid_all.npy")


# ------------------------------------------------------------
# CPU USAGE
# ------------------------------------------------------------

CPU_USAGE = 20


# ------------------------------------------------------------
# LARGE DISPLACEMENT WARNING
# ------------------------------------------------------------

LARGE_DISPLACEMENT_THRESHOLD = 10.0

MAX_WARNING_ATOMS = 10


# ============================================================
# CPU LIMIT
# ============================================================

def limit_cpu_usage(cpu_usage):
    """
    Limit the number of CPUs available to this process.

    cpu_usage:
        Percentage of available CPUs to allow.

        Example:
            50 -> approximately half the CPUs

        None:
            Do not modify CPU affinity.
    """

    if cpu_usage is None:

        print(
            "CPU limit: disabled"
        )

        return

    if not 0 < cpu_usage <= 100:

        raise ValueError(
            "CPU_USAGE must be between 1 and 100, "
            "or None."
        )

    if not hasattr(
        os,
        "sched_getaffinity"
    ):

        print(
            "WARNING: CPU affinity is not supported "
            "on this system."
        )

        print(
            "CPU limit will not be applied."
        )

        return

    available_cpus = sorted(
        os.sched_getaffinity(0)
    )

    n_available = len(
        available_cpus
    )

    n_allowed = max(
        1,
        int(
            np.ceil(
                n_available
                * cpu_usage
                / 100.0
            )
        )
    )

    n_allowed = min(
        n_allowed,
        n_available
    )

    selected_cpus = set(
        available_cpus[
            :n_allowed
        ]
    )

    os.sched_setaffinity(
        0,
        selected_cpus
    )

    print(
        f"CPU limit: {cpu_usage:.0f}%"
    )

    print(
        f"    Available CPUs: {n_available}"
    )

    print(
        f"    CPUs assigned to process: "
        f"{n_allowed}"
    )

    print(
        f"    CPU IDs: "
        f"{sorted(selected_cpus)}"
    )


# ============================================================
# READ LAMMPS DATA FILE
# ============================================================

def read_lammps_data(
    filename,
    protein_types
):
    """
    Read protein atom IDs, types, and coordinates from a
    LAMMPS data file.

    Protein atoms are identified by atom type 1-23.

    Supported common atom styles:

        atomic:
            id type x y z

        molecular:
            id mol type x y z

        charge:
            id type q x y z

        full:
            id mol type q x y z

    Returns
    -------
    protein_ids
        Atom IDs of protein atoms.

    protein_coordinates
        Coordinates of protein atoms.

    protein_types_found
        Atom types of protein atoms.
    """

    if not filename.exists():

        raise FileNotFoundError(
            f"Reference data file not found: "
            f"{filename}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        f"Reading reference structure: {filename}"
    )

    print(
        "=" * 70
    )

    atom_style = None
    atom_start = None

    with open(
        filename,
        "r"
    ) as f:

        lines = f.readlines()

    # ========================================================
    # FIND ATOMS SECTION
    # ========================================================

    for i, raw_line in enumerate(lines):

        line = raw_line.strip()

        if re.match(
            r"^Atoms\b",
            line,
            re.IGNORECASE
        ):

            match = re.search(
                r"Atoms\s*#\s*(\S+)",
                line,
                re.IGNORECASE
            )

            if match:

                atom_style = (
                    match.group(1)
                    .lower()
                )

            atom_start = i + 1

            break

    if atom_start is None:

        raise RuntimeError(
            f"Could not find 'Atoms' section in "
            f"{filename}"
        )

    print(
        f"    Detected atom style: "
        f"{atom_style if atom_style else 'not specified'}"
    )

    # ========================================================
    # READ ATOMS
    # ========================================================

    atom_records = []

    started_reading_atoms = False

    for raw_line in lines[atom_start:]:

        line = raw_line.strip()

        if not line:

            if started_reading_atoms:
                break

            continue

        if line.startswith("#"):
            continue

        # ----------------------------------------------------
        # Stop at next LAMMPS data section
        # ----------------------------------------------------

        first_word = line.split()[0]

        if first_word in {
            "Velocities",
            "Bonds",
            "Angles",
            "Dihedrals",
            "Impropers",
            "PairIJ",
            "Pair",
            "Bond",
            "Angle",
            "Dihedral",
            "Improper",
            "Molecules"
        }:

            if started_reading_atoms:
                break

        fields = line.split()

        # Remove inline comment
        if "#" in fields:

            fields = fields[
                :fields.index("#")
            ]

        if len(fields) < 5:
            continue

        # ----------------------------------------------------
        # Atom ID
        # ----------------------------------------------------

        try:

            atom_id = int(
                fields[0]
            )

        except ValueError:

            if started_reading_atoms:
                break

            continue

        # ====================================================
        # DETERMINE COLUMN POSITIONS
        # ====================================================

        if atom_style == "atomic":

            type_index = 1
            x_index = 2
            y_index = 3
            z_index = 4

        elif atom_style == "molecular":

            type_index = 2
            x_index = 3
            y_index = 4
            z_index = 5

        elif atom_style == "charge":

            type_index = 1
            x_index = 3
            y_index = 4
            z_index = 5

        elif atom_style == "full":

            type_index = 2
            x_index = 4
            y_index = 5
            z_index = 6

        else:

            # ------------------------------------------------
            # Infer style if "Atoms # ..." is not present.
            # ------------------------------------------------

            if len(fields) == 5:

                # id type x y z
                type_index = 1
                x_index = 2
                y_index = 3
                z_index = 4

            elif len(fields) == 6:

                # Assume molecular:
                # id mol type x y z
                type_index = 2
                x_index = 3
                y_index = 4
                z_index = 5

            elif len(fields) >= 7:

                # Assume full:
                # id mol type q x y z
                type_index = 2
                x_index = 4
                y_index = 5
                z_index = 6

            else:

                raise RuntimeError(
                    f"Could not determine atom style "
                    f"from line:\n{line}"
                )

        # ====================================================
        # READ TYPE
        # ====================================================

        try:

            atom_type = int(
                fields[type_index]
            )

        except (
            ValueError,
            IndexError
        ):

            raise RuntimeError(
                f"Could not read atom type from line:\n"
                f"{line}"
            )

        started_reading_atoms = True

        # ====================================================
        # SELECT PROTEIN TYPES 1-23
        # ====================================================

        if atom_type not in protein_types:
            continue

        # ====================================================
        # READ COORDINATES
        # ====================================================

        try:

            x = float(
                fields[x_index]
            )

            y = float(
                fields[y_index]
            )

            z = float(
                fields[z_index]
            )

        except (
            ValueError,
            IndexError
        ):

            raise RuntimeError(
                f"Could not read coordinates from line:\n"
                f"{line}"
            )

        atom_records.append(
            (
                atom_id,
                atom_type,
                x,
                y,
                z
            )
        )

    # ========================================================
    # CHECK RESULT
    # ========================================================

    if not atom_records:

        raise RuntimeError(
            f"No protein atoms with types "
            f"{sorted(protein_types)} "
            f"were found in {filename}"
        )

    # ========================================================
    # SORT BY ATOM ID
    # ========================================================

    atom_records.sort(
        key=lambda x: x[0]
    )

    protein_ids = np.array(
        [
            record[0]
            for record in atom_records
        ],
        dtype=int
    )

    protein_types_found = np.array(
        [
            record[1]
            for record in atom_records
        ],
        dtype=int
    )

    protein_coordinates = np.array(
        [
            [
                record[2],
                record[3],
                record[4]
            ]
            for record in atom_records
        ],
        dtype=float
    )

    # ========================================================
    # CHECK DUPLICATE IDS
    # ========================================================

    if (
        len(np.unique(protein_ids))
        != len(protein_ids)
    ):

        raise RuntimeError(
            f"Duplicate atom IDs found among protein atoms "
            f"in {filename}"
        )

    # ========================================================
    # PRINT INFORMATION
    # ========================================================

    print()
    print(
        f"    Protein atoms found: "
        f"{len(protein_ids)}"
    )

    print(
        f"    Protein atom types found: "
        f"{sorted(np.unique(protein_types_found))}"
    )

    print(
        f"    Protein atom ID range: "
        f"{protein_ids.min()} - "
        f"{protein_ids.max()}"
    )

    return (
        protein_ids,
        protein_coordinates,
        protein_types_found
    )


# ============================================================
# LOAD RMSD ATOM IDS
# ============================================================

def load_backbone_ids(
    filename
):
    """
    Load atom IDs from resid_all.npy.

    These IDs define the subset of protein atoms that should
    actually be used for the RMSD calculation.
    """

    if not filename.exists():

        raise FileNotFoundError(
            f"RMSD atom ID file not found: "
            f"{filename}"
        )

    backbone_ids = np.load(
        filename
    )

    backbone_ids = np.asarray(
        backbone_ids,
        dtype=int
    ).flatten()

    if len(backbone_ids) == 0:

        raise RuntimeError(
            f"{filename} contains no atom IDs."
        )

    if (
        len(np.unique(backbone_ids))
        != len(backbone_ids)
    ):

        raise RuntimeError(
            f"{filename} contains duplicate atom IDs."
        )

    print()
    print(
        f"Loaded {len(backbone_ids)} RMSD atom IDs "
        f"from {filename}"
    )

    print(
        f"    First IDs: "
        f"{backbone_ids[:10]}"
    )

    return backbone_ids


# ============================================================
# SELECT COORDINATES BY ATOM ID
# ============================================================

def select_coordinates_by_id(
    coordinates,
    atom_ids,
    requested_ids,
    timestep
):
    """
    Select coordinates using atom IDs.

    Returned coordinates are in exactly the same order
    as requested_ids.
    """

    id_to_index = {
        int(atom_id): i
        for i, atom_id in enumerate(atom_ids)
    }

    missing_ids = [
        int(atom_id)
        for atom_id in requested_ids
        if int(atom_id) not in id_to_index
    ]

    if missing_ids:

        raise RuntimeError(
            f"Missing atom IDs at timestep {timestep}.\n"
            f"Number missing: {len(missing_ids)}\n"
            f"First missing IDs: "
            f"{missing_ids[:20]}"
        )

    indices = [
        id_to_index[int(atom_id)]
        for atom_id in requested_ids
    ]

    return coordinates[
        indices
    ]


# ============================================================
# KABSCH
# ============================================================

def kabsch_rotation(
    current_positions,
    target_positions
):
    """
    Compute the optimal rotation matrix that aligns
    current_positions to target_positions.
    """

    H = (
        current_positions.T
        @ target_positions
    )

    U, S, Vt = np.linalg.svd(H)

    R = (
        Vt.T
        @ U.T
    )

    # --------------------------------------------------------
    # Prevent reflection
    # --------------------------------------------------------

    if np.linalg.det(R) < 0:

        Vt[-1, :] *= -1

        R = (
            Vt.T
            @ U.T
        )

    return R


# ============================================================
# RMSD
# ============================================================

def calculate_rmsd(
    current,
    reference
):
    """
    Calculate RMSD between two centered and aligned
    coordinate sets.
    """

    difference = (
        current
        - reference
    )

    squared_distances = np.sum(
        difference ** 2,
        axis=1
    )

    return np.sqrt(
        np.mean(
            squared_distances
        )
    )


# ============================================================
# LARGE DISPLACEMENT CHECK
# ============================================================

def check_large_displacements(
    aligned,
    reference,
    atom_ids,
    timestep,
    threshold=LARGE_DISPLACEMENT_THRESHOLD
):
    """
    Check per-atom distances between aligned current
    coordinates and reference coordinates.
    """

    atom_displacements = np.linalg.norm(
        aligned - reference,
        axis=1
    )

    large_atoms = np.where(
        atom_displacements > threshold
    )[0]

    if len(large_atoms) > 0:

        print()
        print(
            "!" * 70
        )

        print(
            f"WARNING: {len(large_atoms)} RMSD atoms "
            f"have displacements > "
            f"{threshold:.1f} Å "
            f"at timestep {timestep}"
        )

        print(
            "    Largest atom displacements:"
        )

        worst_atoms = large_atoms[
            np.argsort(
                atom_displacements[
                    large_atoms
                ]
            )[::-1]
        ]

        for index in worst_atoms[
            :MAX_WARNING_ATOMS
        ]:

            print(
                f"        atom ID "
                f"{atom_ids[index]:8d}: "
                f"{atom_displacements[index]:12.3f} Å"
            )

        print(
            "!" * 70
        )

        print()

    return atom_displacements


# ============================================================
# READ LAMMPS TRAJECTORY
# ============================================================

def read_trajectory(
    filename
):
    """
    Read a LAMMPS trajectory.

    Expected atom columns include:

        id type x y z

    Atoms are sorted by ID in every frame.
    """

    frames = []

    print(
        f"    Reading trajectory: {filename}"
    )

    with open(
        filename,
        "r"
    ) as f:

        while True:

            line = f.readline()

            if not line:
                break

            if not line.startswith(
                "ITEM: TIMESTEP"
            ):
                continue

            timestep = int(
                f.readline().strip()
            )

            # =================================================
            # NUMBER OF ATOMS
            # =================================================

            line = f.readline()

            if not line.startswith(
                "ITEM: NUMBER OF ATOMS"
            ):

                raise RuntimeError(
                    f"Unexpected format at timestep "
                    f"{timestep}"
                )

            n_atoms = int(
                f.readline().strip()
            )

            # =================================================
            # BOX BOUNDS
            # =================================================

            line = f.readline()

            if not line.startswith(
                "ITEM: BOX BOUNDS"
            ):

                raise RuntimeError(
                    f"Expected BOX BOUNDS at timestep "
                    f"{timestep}"
                )

            f.readline()
            f.readline()
            f.readline()

            # =================================================
            # ATOM HEADER
            # =================================================

            line = f.readline()

            if not line.startswith(
                "ITEM: ATOMS"
            ):

                raise RuntimeError(
                    f"Expected ATOMS section at timestep "
                    f"{timestep}"
                )

            columns = (
                line.strip()
                .split()[2:]
            )

            try:

                id_index = columns.index(
                    "id"
                )

                x_index = columns.index(
                    "x"
                )

                y_index = columns.index(
                    "y"
                )

                z_index = columns.index(
                    "z"
                )

            except ValueError:

                raise RuntimeError(
                    f"Could not find id/x/y/z columns.\n"
                    f"Available columns: {columns}"
                )

            # =================================================
            # READ ATOMS
            # =================================================

            atom_data = []

            for _ in range(n_atoms):

                fields = f.readline().split()

                if len(fields) < len(columns):

                    raise RuntimeError(
                        f"Invalid atom line at timestep "
                        f"{timestep}"
                    )

                atom_id = int(
                    fields[id_index]
                )

                x = float(
                    fields[x_index]
                )

                y = float(
                    fields[y_index]
                )

                z = float(
                    fields[z_index]
                )

                atom_data.append(
                    (
                        atom_id,
                        x,
                        y,
                        z
                    )
                )

            # =================================================
            # SORT BY ATOM ID
            # =================================================

            atom_data.sort(
                key=lambda x: x[0]
            )

            atom_ids = [
                atom[0]
                for atom in atom_data
            ]

            # =================================================
            # CHECK IDS
            # =================================================

            if len(set(atom_ids)) != n_atoms:

                raise RuntimeError(
                    f"Duplicate atom IDs at timestep "
                    f"{timestep}"
                )

            coordinates = np.array(
                [
                    [
                        atom[1],
                        atom[2],
                        atom[3]
                    ]
                    for atom in atom_data
                ],
                dtype=float
            )

            frames.append({
                "timestep": timestep,
                "atom_ids": atom_ids,
                "coordinates": coordinates
            })

    print(
        f"    Read {len(frames)} frames"
    )

    return frames


# ============================================================
# PROCESS ONE TRAJECTORY
# ============================================================

def process_trajectory(
    filename,
    backbone_ids,
    reference
):
    """
    Calculate RMSD relative to dfr_start.data.

    backbone_ids contains the final RMSD atom IDs obtained
    from:

        resid_all.npy
        AND
        protein atoms (types 1-23) in dfr_start.data
    """

    frames = read_trajectory(
        filename
    )

    if not frames:

        raise RuntimeError(
            f"No frames found in {filename}"
        )

    # ========================================================
    # CENTER REFERENCE
    # ========================================================

    reference_centered = (
        reference
        - reference.mean(axis=0)
    )

    print()
    print(
        f"    Reference structure: "
        f"{REFERENCE_DATA_FILE}"
    )

    print(
        f"    RMSD atoms used: "
        f"{len(backbone_ids)}"
    )

    results = []

    # ========================================================
    # PROCESS EVERY FRAME
    # ========================================================

    for frame_number, frame in enumerate(
        frames
    ):

        timestep = frame[
            "timestep"
        ]

        current_coordinates_all = (
            frame["coordinates"]
        )

        current_ids = (
            frame["atom_ids"]
        )

        # ====================================================
        # MATCH RMSD ATOMS BY ID
        # ====================================================

        current = select_coordinates_by_id(
            current_coordinates_all,
            current_ids,
            backbone_ids,
            timestep
        )

        # ====================================================
        # CENTER CURRENT
        # ====================================================

        current_centered = (
            current
            - current.mean(axis=0)
        )

        # ====================================================
        # KABSCH ALIGNMENT
        #
        # current -> dfr_start.data reference
        # ====================================================

        R = kabsch_rotation(
            current_centered,
            reference_centered
        )

        # ====================================================
        # APPLY ROTATION
        # ====================================================

        aligned = (
            current_centered
            @ R.T
        )

        # ====================================================
        # LARGE DISPLACEMENT CHECK
        # ====================================================

        atom_displacements = (
            check_large_displacements(
                aligned,
                reference_centered,
                backbone_ids,
                timestep
            )
        )

        # ====================================================
        # RMSD
        # ====================================================

        rmsd = calculate_rmsd(
            aligned,
            reference_centered
        )

        # ====================================================
        # PRINT DIAGNOSTICS
        # ====================================================

        print(
            f"    Frame {frame_number:6d} "
            f"timestep {timestep:10d} "
            f"RMSD = {rmsd:10.4f} Å "
            f"max displacement = "
            f"{atom_displacements.max():10.4f} Å"
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

        if re.fullmatch(
            r"eval_\d+",
            path.name
        ):

            eval_dirs.append(
                path
            )

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

    # ========================================================
    # CPU LIMIT
    # ========================================================

    limit_cpu_usage(
        CPU_USAGE
    )

    # ========================================================
    # CHECK MODEL DIRECTORY
    # ========================================================

    if not MODEL_DIR.exists():

        raise FileNotFoundError(
            f"Model directory not found: "
            f"{MODEL_DIR}"
        )

    # ========================================================
    # CHECK REFERENCE FILE
    # ========================================================

    if not REFERENCE_DATA_FILE.exists():

        raise FileNotFoundError(
            f"Reference data file not found: "
            f"{REFERENCE_DATA_FILE}"
        )

    # ========================================================
    # LOAD RMSD IDS FROM NPY
    # ========================================================

    backbone_ids_from_npy = load_backbone_ids(
        BACKBONE_IDS_FILE
    )

    # ========================================================
    # LOAD PROTEIN FROM dfr_start.data
    # ========================================================

    (
        protein_ids,
        protein_coordinates,
        protein_types
    ) = read_lammps_data(
        REFERENCE_DATA_FILE,
        PROTEIN_TYPES
    )

    # ========================================================
    # RESTRICT NPY IDS TO PROTEIN ATOMS
    # ========================================================
    #
    # Only IDs that are:
    #
    #   1. in resid_all.npy
    #   2. protein atoms (types 1-23)
    #
    # are used for RMSD.
    # ========================================================

    protein_id_set = set(
        protein_ids.tolist()
    )

    invalid_ids = [
        int(atom_id)
        for atom_id in backbone_ids_from_npy
        if int(atom_id) not in protein_id_set
    ]

    if invalid_ids:

        print()
        print(
            "WARNING:"
        )

        print(
            f"    {len(invalid_ids)} IDs from "
            f"{BACKBONE_IDS_FILE} are not protein atoms "
            f"(types 1-23) in {REFERENCE_DATA_FILE}."
        )

        print(
            f"    First invalid IDs: "
            f"{invalid_ids[:20]}"
        )

        print(
            "    These IDs will be excluded."
        )

    # --------------------------------------------------------
    # Final RMSD atom IDs
    # --------------------------------------------------------

    backbone_ids = np.array(
        [
            int(atom_id)
            for atom_id in backbone_ids_from_npy
            if int(atom_id) in protein_id_set
        ],
        dtype=int
    )

    if len(backbone_ids) == 0:

        raise RuntimeError(
            "No IDs from resid_all.npy correspond to "
            "protein atoms (types 1-23) in "
            f"{REFERENCE_DATA_FILE}."
        )

    # ========================================================
    # SELECT REFERENCE COORDINATES
    # ========================================================

    reference = select_coordinates_by_id(
        protein_coordinates,
        protein_ids,
        backbone_ids,
        "dfr_start.data"
    )

    # ========================================================
    # PRINT FINAL SELECTION
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "RMSD ATOM SELECTION"
    )

    print(
        "=" * 70
    )

    print(
        f"    Total protein atoms in data file: "
        f"{len(protein_ids)}"
    )

    print(
        f"    IDs in resid_all.npy: "
        f"{len(backbone_ids_from_npy)}"
    )

    print(
        f"    Final RMSD atoms: "
        f"{len(backbone_ids)}"
    )

    print(
        f"    Protein types allowed: "
        f"1-23"
    )

    print(
        f"    Water type excluded: "
        f"24"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # FIND EVAL DIRECTORIES
    # ========================================================

    eval_dirs = find_eval_directories()

    if not eval_dirs:

        raise RuntimeError(
            f"No eval_* directories found in "
            f"{MODEL_DIR}"
        )

    print()
    print(
        f"Found {len(eval_dirs)} evaluation directories."
    )

    # ========================================================
    # ALL RESULTS
    # ========================================================

    all_results = []

    # ========================================================
    # PROCESS EACH EVALUATION
    # ========================================================

    for eval_dir in eval_dirs:

        trajectory = (
            eval_dir
            / TRAJECTORY_NAME
        )

        if not trajectory.exists():

            print(
                f"WARNING: {trajectory} does not exist "
                "-- skipping."
            )

            continue

        print()
        print(
            "=" * 70
        )

        print(
            f"Processing {eval_dir.name}"
        )

        print(
            "=" * 70
        )

        results = process_trajectory(
            trajectory,
            backbone_ids,
            reference
        )

        # ====================================================
        # PER-EVALUATION OUTPUT
        # ====================================================

        output_file = (
            eval_dir
            / "rmsd.csv"
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

        # ====================================================
        # COMBINED RESULTS
        # ====================================================

        for r in results:

            all_results.append([
                eval_dir.name,
                r["frame"],
                r["timestep"],
                r["rmsd"]
            ])

    # ========================================================
    # SORT RESULTS
    # ========================================================

    all_results.sort(
        key=lambda x: (
            int(
                x[0].split("_")[1]
            ),
            int(x[1])
        )
    )

    # ========================================================
    # COMBINED CSV
    # ========================================================

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

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "Done."
    )

    print(
        f"Reference structure: "
        f"{REFERENCE_DATA_FILE}"
    )

    print(
        f"Protein atom types: "
        f"1-23"
    )

    print(
        f"Water type excluded: "
        f"24"
    )

    print(
        f"IDs loaded from: "
        f"{BACKBONE_IDS_FILE}"
    )

    print(
        f"Final RMSD atoms: "
        f"{len(backbone_ids)}"
    )

    print(
        f"Combined output: "
        f"{COMBINED_OUTPUT}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()
