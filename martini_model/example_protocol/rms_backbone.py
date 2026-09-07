#!/usr/bin/env python3

from pathlib import Path
import csv
import re

import numpy as np


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path("lammps")
MODEL_DIR = BASE_DIR / "model_400000"

# ------------------------------------------------------------
# LAMMPS DATA FILE
# ------------------------------------------------------------
#
# This file provides the FIXED REFERENCE STRUCTURE.
#
# Protein atoms are represented by atom types 1-23.
# Atom type 24 is water and is ignored.
#

DATA_FILE = BASE_DIR / "dfr_start.data"

# ------------------------------------------------------------
# TRAJECTORY
# ------------------------------------------------------------

TRAJECTORY_NAME = "protein.lammpstrj"

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

COMBINED_OUTPUT = MODEL_DIR / "rmsd_all_equi.csv"

# ------------------------------------------------------------
# RMSD ATOM IDS
# ------------------------------------------------------------
#
# resid_all.npy defines the atoms that are used for the
# RMSD calculation.
#

BB_IDS_FILE = Path("resid_all.npy")

# ------------------------------------------------------------
# NUMBER OF INITIAL FRAMES TO SKIP
# ------------------------------------------------------------

SKIP_FRAMES = 0

# ------------------------------------------------------------
# LARGE DISPLACEMENT WARNING
# ------------------------------------------------------------
#
# A warning is printed if any atom differs from the reference
# by more than this distance AFTER alignment.
#

LARGE_DISPLACEMENT_THRESHOLD = 10.0

# Number of worst atoms to print when a warning occurs.
MAX_WARNING_ATOMS = 10


# ============================================================
# KABSCH
# ============================================================

def kabsch_rotation(
    current_positions,
    target_positions
):

    """
    Calculate the optimal rotation matrix that aligns
    current_positions onto target_positions.

    Parameters
    ----------
    current_positions : ndarray
        Shape (N, 3)

    target_positions : ndarray
        Shape (N, 3)

    Returns
    -------
    R : ndarray
        Shape (3, 3)
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


def align_structure(
    current,
    target
):

    """
    Align current coordinates onto target coordinates
    using translation + Kabsch rotation.

    Both structures are centered on their respective
    centers of geometry before calculating the rotation.

    Parameters
    ----------
    current : ndarray
        Shape (N, 3)

    target : ndarray
        Shape (N, 3)

    Returns
    -------
    aligned : ndarray
        Shape (N, 3)
    """

    # --------------------------------------------------------
    # Center current structure
    # --------------------------------------------------------

    current_centered = (
        current
        - current.mean(axis=0)
    )

    # --------------------------------------------------------
    # Center target structure
    # --------------------------------------------------------

    target_centered = (
        target
        - target.mean(axis=0)
    )

    # --------------------------------------------------------
    # Kabsch rotation
    # --------------------------------------------------------

    R = kabsch_rotation(
        current_centered,
        target_centered
    )

    # --------------------------------------------------------
    # Apply rotation
    # --------------------------------------------------------

    aligned = (
        current_centered
        @ R.T
    )

    return aligned


# ============================================================
# READ LAMMPS DATA FILE
# ============================================================

def read_lammps_data(
    filename
):

    """
    Read the LAMMPS data file and extract the protein atoms.

    Protein:
        atom types 1-23

    Water:
        atom type 24 -> ignored

    The function returns ALL protein atoms. The final set of
    atoms used for RMSD is selected later using resid_all.npy.

    The parser assumes a standard LAMMPS Atoms section in
    which:

        field 0 = atom ID
        field 2 = atom type

    and the final three fields are x, y, z.
    """

    if not filename.exists():

        raise FileNotFoundError(
            f"LAMMPS data file not found: "
            f"{filename}"
        )

    print()
    print(
        "Reading reference structure:"
    )

    print(
        f"    {filename}"
    )

    with open(
        filename,
        "r"
    ) as f:

        lines = f.readlines()

    # ========================================================
    # FIND ATOMS SECTION
    # ========================================================

    atoms_start = None

    for i, line in enumerate(lines):

        stripped = line.strip()

        if (
            stripped == "Atoms"
            or stripped.startswith("Atoms #")
        ):

            atoms_start = i + 1

            break

    if atoms_start is None:

        raise RuntimeError(
            f"Could not find an 'Atoms' section in "
            f"{filename}"
        )

    # ========================================================
    # READ ATOMS
    # ========================================================

    atom_records = []

    for line in lines[atoms_start:]:

        stripped = line.strip()

        # ----------------------------------------------------
        # Skip initial blank lines after Atoms header
        # ----------------------------------------------------

        if not stripped:

            if atom_records:
                break

            continue

        # ----------------------------------------------------
        # Stop when another LAMMPS section begins
        # ----------------------------------------------------

        if re.match(
            r"^(Velocities|Bonds|Angles|Dihedrals|"
            r"Impropers|Masses|Pair Coeffs|Bond Coeffs|"
            r"Angle Coeffs|Dihedral Coeffs|"
            r"Improper Coeffs)\b",
            stripped
        ):

            break

        fields = stripped.split()

        # ----------------------------------------------------
        # Minimum expected fields
        # ----------------------------------------------------

        if len(fields) < 6:
            continue

        try:

            atom_id = int(
                fields[0]
            )

            atom_type = int(
                fields[2]
            )

            x = float(
                fields[-3]
            )

            y = float(
                fields[-2]
            )

            z = float(
                fields[-1]
            )

        except ValueError:

            # Ignore lines that are not atom records
            continue

        atom_records.append(
            (
                atom_id,
                atom_type,
                x,
                y,
                z
            )
        )

    if not atom_records:

        raise RuntimeError(
            f"No atom records found in the Atoms section "
            f"of {filename}"
        )

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    atom_ids = np.asarray(
        [
            record[0]
            for record in atom_records
        ],
        dtype=int
    )

    atom_types = np.asarray(
        [
            record[1]
            for record in atom_records
        ],
        dtype=int
    )

    coordinates = np.asarray(
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
        len(np.unique(atom_ids))
        != len(atom_ids)
    ):

        raise RuntimeError(
            "Duplicate atom IDs found in the LAMMPS "
            "data file."
        )

    # ========================================================
    # SELECT PROTEIN
    # ========================================================

    protein_mask = (
        (atom_types >= 1)
        &
        (atom_types <= 23)
    )

    water_mask = (
        atom_types == 24
    )

    n_total = len(atom_ids)

    n_protein = np.sum(
        protein_mask
    )

    n_water = np.sum(
        water_mask
    )

    print(
        f"    Total atoms in data file: "
        f"{n_total}"
    )

    print(
        f"    Protein atoms (types 1-23): "
        f"{n_protein}"
    )

    print(
        f"    Water atoms (type 24): "
        f"{n_water}"
    )

    if n_protein == 0:

        raise RuntimeError(
            "No protein atoms with types 1-23 "
            "were found in the data file."
        )

    # --------------------------------------------------------
    # Keep only protein
    # --------------------------------------------------------

    atom_ids = atom_ids[
        protein_mask
    ]

    atom_types = atom_types[
        protein_mask
    ]

    coordinates = coordinates[
        protein_mask
    ]

    # ========================================================
    # SORT BY ATOM ID
    # ========================================================

    order = np.argsort(
        atom_ids
    )

    atom_ids = atom_ids[
        order
    ]

    atom_types = atom_types[
        order
    ]

    coordinates = coordinates[
        order
    ]

    # ========================================================
    # PRINT TYPE INFORMATION
    # ========================================================

    unique_types, counts = np.unique(
        atom_types,
        return_counts=True
    )

    print(
        "    Protein atom types:"
    )

    for atom_type, count in zip(
        unique_types,
        counts
    ):

        print(
            f"        type {atom_type:2d}: "
            f"{count}"
        )

    return {
        "atom_ids": atom_ids,
        "atom_types": atom_types,
        "coordinates": coordinates
    }


# ============================================================
# READ LAMMPS TRAJECTORY
# ============================================================

def read_trajectory(
    filename
):

    """
    Read a LAMMPS trajectory.

    The trajectory must contain:

        ITEM: TIMESTEP
        ITEM: NUMBER OF ATOMS
        ITEM: BOX BOUNDS
        ITEM: ATOMS ...

    The ATOMS header must contain:

        id
        type
        x
        y
        z

    Atom ordering does not need to be identical between
    frames because atoms are matched using their IDs.
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

            # =================================================
            # TIMESTEP
            # =================================================

            timestep_line = f.readline()

            if not timestep_line:

                raise RuntimeError(
                    "Unexpected end of file while "
                    "reading timestep."
                )

            timestep = int(
                timestep_line.strip()
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
                    f"{timestep}: expected "
                    f"ITEM: NUMBER OF ATOMS"
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

            # Standard 3D orthogonal box
            f.readline()
            f.readline()
            f.readline()

            # =================================================
            # ATOMS HEADER
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

            # -------------------------------------------------
            # Required columns
            # -------------------------------------------------

            try:

                id_index = columns.index(
                    "id"
                )

                type_index = columns.index(
                    "type"
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
                    f"Could not find required "
                    f"id/type/x/y/z columns at timestep "
                    f"{timestep}.\n"
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
                        f"{timestep}."
                    )

                atom_id = int(
                    fields[id_index]
                )

                atom_type = int(
                    fields[type_index]
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
                        atom_type,
                        x,
                        y,
                        z
                    )
                )

            # =================================================
            # SORT BY ATOM ID
            # =================================================

            atom_data.sort(
                key=lambda record: record[0]
            )

            atom_ids = np.asarray(
                [
                    record[0]
                    for record in atom_data
                ],
                dtype=int
            )

            atom_types = np.asarray(
                [
                    record[1]
                    for record in atom_data
                ],
                dtype=int
            )

            coordinates = np.asarray(
                [
                    [
                        record[2],
                        record[3],
                        record[4]
                    ]
                    for record in atom_data
                ],
                dtype=float
            )

            # =================================================
            # CHECK DUPLICATE IDS
            # =================================================

            if (
                len(np.unique(atom_ids))
                != n_atoms
            ):

                raise RuntimeError(
                    f"Duplicate atom IDs at timestep "
                    f"{timestep}"
                )

            frames.append({
                "timestep": timestep,
                "atom_ids": atom_ids,
                "atom_types": atom_types,
                "coordinates": coordinates
            })

    print(
        f"    Read {len(frames)} frames"
    )

    return frames


# ============================================================
# SELECT ATOMS BY ID
# ============================================================

def select_atoms_by_id(
    coordinates,
    atom_ids,
    requested_ids
):

    """
    Select coordinates corresponding to requested atom IDs.

    The returned coordinates are in exactly the same order
    as requested_ids.
    """

    id_to_index = {
        atom_id: i
        for i, atom_id in enumerate(atom_ids)
    }

    missing_ids = [
        atom_id
        for atom_id in requested_ids
        if atom_id not in id_to_index
    ]

    if missing_ids:

        raise RuntimeError(
            f"Missing requested atom IDs.\n"
            f"Number missing: {len(missing_ids)}\n"
            f"First missing IDs: "
            f"{missing_ids[:20]}"
        )

    indices = [
        id_to_index[atom_id]
        for atom_id in requested_ids
    ]

    return coordinates[
        indices
    ]


# ============================================================
# SELECT RMSD ATOMS FROM REFERENCE
# ============================================================

def select_reference_rmsd_atoms(
    reference,
    bb_ids
):

    """
    Select the atoms specified by resid_all.npy from the
    fixed LAMMPS data-file reference.

    Every resid_all.npy atom must belong to the protein
    (types 1-23).
    """

    reference_ids = (
        reference["atom_ids"]
    )

    reference_types = (
        reference["atom_types"]
    )

    reference_coordinates = (
        reference["coordinates"]
    )

    # --------------------------------------------------------
    # Map atom ID -> reference index
    # --------------------------------------------------------

    id_to_index = {
        atom_id: i
        for i, atom_id in enumerate(
            reference_ids
        )
    }

    # --------------------------------------------------------
    # Check that every RMSD ID exists
    # --------------------------------------------------------

    missing_ids = [
        atom_id
        for atom_id in bb_ids
        if atom_id not in id_to_index
    ]

    if missing_ids:

        raise RuntimeError(
            "Some IDs from resid_all.npy are not present "
            "as protein atoms (types 1-23) in the LAMMPS "
            "data file.\n"
            f"Number missing: {len(missing_ids)}\n"
            f"First missing IDs: {missing_ids[:20]}"
        )

    # --------------------------------------------------------
    # Select reference coordinates
    # --------------------------------------------------------

    indices = [
        id_to_index[atom_id]
        for atom_id in bb_ids
    ]

    rmsd_reference = (
        reference_coordinates[
            indices
        ]
    )

    rmsd_types = (
        reference_types[
            indices
        ]
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    invalid_types = (
        ~(
            (rmsd_types >= 1)
            &
            (rmsd_types <= 23)
        )
    )

    if np.any(invalid_types):

        bad_ids = bb_ids[
            invalid_types
        ]

        bad_types = rmsd_types[
            invalid_types
        ]

        raise RuntimeError(
            "resid_all.npy contains atoms that are not "
            "protein atoms.\n"
            f"Bad IDs: {bad_ids[:20]}\n"
            f"Bad types: {bad_types[:20]}"
        )

    return rmsd_reference


# ============================================================
# RMSD
# ============================================================

def calculate_rmsd(
    current,
    reference
):

    """
    Calculate RMSD between two structures.

    Both structures must already be aligned.
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

    Returns
    -------
    atom_displacements : ndarray
        Per-atom displacement in Angstrom.
    """

    # --------------------------------------------------------
    # Calculate distance for every atom
    # --------------------------------------------------------

    atom_displacements = np.linalg.norm(
        aligned - reference,
        axis=1
    )

    # --------------------------------------------------------
    # Find atoms above threshold
    # --------------------------------------------------------

    large_atoms = np.where(
        atom_displacements > threshold
    )[0]

    if len(large_atoms) > 0:

        print()
        print(
            "!" * 70
        )

        print(
            f"WARNING: {len(large_atoms)} atoms have "
            f"displacements > {threshold:.1f} Å "
            f"at timestep {timestep}"
        )

        print(
            "    Largest atom displacements:"
        )

        # Sort by displacement, largest first
        worst_atoms = large_atoms[
            np.argsort(
                atom_displacements[large_atoms]
            )[::-1]
        ]

        for index in worst_atoms[:MAX_WARNING_ATOMS]:

            print(
                f"        atom ID {atom_ids[index]:8d}: "
                f"{atom_displacements[index]:12.3f} Å"
            )

        print(
            "!" * 70
        )
        print()

    return atom_displacements


# ============================================================
# PROCESS ONE TRAJECTORY
# ============================================================

def process_trajectory(
    filename,
    reference_ids,
    reference_coordinates,
    reference_types
):

    """
    Process one trajectory using the LAMMPS data file as the
    fixed reference.

    Only atoms listed in resid_all.npy are used for RMSD.
    """

    frames = read_trajectory(
        filename
    )

    if len(frames) <= SKIP_FRAMES:

        raise RuntimeError(
            f"Trajectory contains only "
            f"{len(frames)} frames, but "
            f"SKIP_FRAMES = {SKIP_FRAMES}"
        )

    # ========================================================
    # CENTER FIXED REFERENCE
    # ========================================================

    reference_centered = (
        reference_coordinates
        - reference_coordinates.mean(
            axis=0
        )
    )

    # ========================================================
    # PROCESS FRAMES
    # ========================================================

    results = []

    for i, frame_number in enumerate(
        range(
            SKIP_FRAMES,
            len(frames)
        )
    ):

        frame = frames[
            frame_number
        ]

        timestep = (
            frame["timestep"]
        )

        current_ids = (
            frame["atom_ids"]
        )

        current_types = (
            frame["atom_types"]
        )

        current_coordinates = (
            frame["coordinates"]
        )

        # ====================================================
        # SELECT SAME ATOMS AS resid_all.npy
        # ====================================================

        current = select_atoms_by_id(
            current_coordinates,
            current_ids,
            reference_ids
        )

        # ====================================================
        # CHECK TYPES
        # ====================================================

        id_to_index = {
            atom_id: j
            for j, atom_id in enumerate(
                current_ids
            )
        }

        current_type_for_rmsd = np.asarray(
            [
                current_types[
                    id_to_index[atom_id]
                ]
                for atom_id in reference_ids
            ],
            dtype=int
        )

        if not np.array_equal(
            current_type_for_rmsd,
            reference_types
        ):

            differing = np.where(
                current_type_for_rmsd
                != reference_types
            )[0]

            first_differences = (
                differing[:20]
            )

            raise RuntimeError(
                f"Atom types changed relative to the "
                f"reference at timestep {timestep}.\n"
                f"First differing atom IDs: "
                f"{reference_ids[first_differences]}\n"
                f"Reference types: "
                f"{reference_types[first_differences]}\n"
                f"Trajectory types: "
                f"{current_type_for_rmsd[first_differences]}"
            )

        # ====================================================
        # ALIGN CURRENT FRAME TO FIXED REFERENCE
        # ====================================================

        aligned = align_structure(
            current,
            reference_centered
        )

        # ====================================================
        # CHECK FOR LARGE ATOM DISPLACEMENTS
        # ====================================================

        atom_displacements = check_large_displacements(
            aligned,
            reference_centered,
            reference_ids,
            timestep
        )

        # ====================================================
        # RMSD
        # ====================================================

        rmsd = calculate_rmsd(
            aligned,
            reference_centered
        )

        # ----------------------------------------------------
        # Print frame diagnostics
        # ----------------------------------------------------

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
    # CHECK MODEL DIRECTORY
    # ========================================================

    if not MODEL_DIR.exists():

        raise FileNotFoundError(
            f"Model directory not found: "
            f"{MODEL_DIR}"
        )

    # ========================================================
    # CHECK DATA FILE
    # ========================================================

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"LAMMPS data file not found: "
            f"{DATA_FILE}"
        )

    # ========================================================
    # CHECK resid_all.npy
    # ========================================================

    if not BB_IDS_FILE.exists():

        raise FileNotFoundError(
            f"RMSD atom ID file not found: "
            f"{BB_IDS_FILE}"
        )

    # ========================================================
    # LOAD resid_all.npy
    # ========================================================

    bb_ids = np.load(
        BB_IDS_FILE
    )

    bb_ids = np.asarray(
        bb_ids,
        dtype=int
    ).flatten()

    if len(bb_ids) == 0:

        raise RuntimeError(
            "resid_all.npy contains no atom IDs."
        )

    # --------------------------------------------------------
    # Check duplicate IDs
    # --------------------------------------------------------

    if (
        len(np.unique(bb_ids))
        != len(bb_ids)
    ):

        raise RuntimeError(
            "resid_all.npy contains duplicate atom IDs."
        )

    print()
    print(
        f"Loaded {len(bb_ids)} atom IDs from "
        f"{BB_IDS_FILE}"
    )

    # ========================================================
    # READ FIXED REFERENCE FROM DATA FILE
    # ========================================================

    reference = read_lammps_data(
        DATA_FILE
    )

    # ========================================================
    # SELECT RMSD ATOMS FROM DATA-FILE REFERENCE
    # ========================================================

    reference_coordinates = (
        select_reference_rmsd_atoms(
            reference,
            bb_ids
        )
    )

    # --------------------------------------------------------
    # Get corresponding atom types
    # --------------------------------------------------------

    reference_id_to_index = {
        atom_id: i
        for i, atom_id in enumerate(
            reference["atom_ids"]
        )
    }

    reference_types = np.asarray(
        [
            reference["atom_types"][
                reference_id_to_index[atom_id]
            ]
            for atom_id in bb_ids
        ],
        dtype=int
    )

    # ========================================================
    # PRINT REFERENCE INFORMATION
    # ========================================================

    print()
    print(
        "Fixed RMSD reference:"
    )

    print(
        f"    Data file: {DATA_FILE}"
    )

    print(
        f"    Total protein atoms: "
        f"{len(reference['atom_ids'])}"
    )

    print(
        f"    RMSD atoms from resid_all.npy: "
        f"{len(bb_ids)}"
    )

    print(
        f"    Protein atom types: 1-23"
    )

    print(
        f"    Water atom type: 24 (ignored)"
    )

    print(
        f"    Skipping first {SKIP_FRAMES} frames"
    )

    print(
        f"    Large displacement warning: "
        f"> {LARGE_DISPLACEMENT_THRESHOLD:.1f} Å"
    )

    print(
        f"    Maximum atoms shown per warning: "
        f"{MAX_WARNING_ATOMS}"
    )

    # ========================================================
    # FIND EVAL DIRECTORIES
    # ========================================================

    eval_dirs = (
        find_eval_directories()
    )

    if not eval_dirs:

        raise RuntimeError(
            f"No eval_* directories found "
            f"in {MODEL_DIR}"
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

            print()
            print(
                f"WARNING: {trajectory} "
                f"does not exist -- skipping."
            )

            continue

        print()
        print(
            "=" * 60
        )

        print(
            f"Processing {eval_dir.name}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Process trajectory
        # ----------------------------------------------------

        results = process_trajectory(
            trajectory,
            bb_ids,
            reference_coordinates,
            reference_types
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

            for result in results:

                writer.writerow([
                    result["frame"],
                    result["timestep"],
                    f"{result['rmsd']:.10f}"
                ])

        print(
            f"    Output: {output_file}"
        )

        # ====================================================
        # ADD TO COMBINED RESULTS
        # ====================================================

        for result in results:

            all_results.append([
                eval_dir.name,
                result["frame"],
                result["timestep"],
                result["rmsd"]
            ])

    # ========================================================
    # SORT COMBINED RESULTS
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
    # WRITE COMBINED CSV
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
        "=" * 60
    )

    print(
        "Done."
    )

    print(
        f"Combined output: "
        f"{COMBINED_OUTPUT}"
    )

    print(
        f"RMSD atoms: "
        f"{len(bb_ids)}"
    )

    print(
        f"Reference: "
        f"{DATA_FILE}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()
