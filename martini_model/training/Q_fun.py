import numpy as np
import subprocess
import shlex
import os

# read data section --------------------------------------------------------

def read_lammpstrj(file_name):

    masses = []

    file = open(file_name, 'r')
    lines = file.readlines()
    write = False
    write_count = 0
    all_positions = []
    for i in range(len(lines)):
        lines[i] = lines[i].split()
        if len(lines[i]) < 2:
            continue
        if lines[i][1] == 'ATOMS':
            write = True
            write_count += 1
            position = []
            continue
        if lines[i][0] == 'ITEM:' and write:
            write = False
            positions = np.array(position)
            # sort elements in positions according to their id (first index of each element)
            positions = positions[positions[:, 0].argsort()]
            # remove first column
            all_positions.append(positions[:, 1:])
            continue
        if write:
            if int(lines[i][1]) < 24:
                if write_count == 1:
                    masses.append(0) # correct masses later ----------------------------!!
                position.append([int(lines[i][0]), float(lines[i][2]), float(lines[i][3]), float(lines[i][4])])


    all_positions = np.array(all_positions)
    masses = np.array(masses)

    return all_positions, masses

def read_data(filename, atom_ids=None, type_cutoff=19, return_ids=False):

    selected = []
    requested_ids = None if atom_ids is None else {int(atom_id) for atom_id in atom_ids}

    with open(filename, 'r') as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            start = i + 1
            break

    if start is None:
        raise ValueError(f"No 'Atoms' section found in file: {filename}")

    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if not parts[0].lstrip("+-").isdigit():
            break
        if len(parts) < 7:
            continue

        atom_id = int(parts[0])
        atom_type = int(parts[2])
        x, y, z = map(float, parts[4:7])

        if requested_ids is not None:
            if atom_id in requested_ids:
                selected.append((atom_id, atom_type, x, y, z))
        elif atom_type < type_cutoff:
            selected.append((atom_id, atom_type, x, y, z))

    selected.sort(key=lambda item: item[0])

    if requested_ids is not None:
        found_ids = {item[0] for item in selected}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            preview = ", ".join(str(atom_id) for atom_id in missing_ids[:10])
            raise ValueError(
                f"Missing {len(missing_ids)} requested atom IDs in {filename}: {preview}"
            )

    ids = np.array([item[0] for item in selected], dtype=np.int64)
    positions = np.array([(item[2], item[3], item[4]) for item in selected], dtype=np.float64)

    if return_ids:
        return ids, positions
    return positions

# save lammps file -------------------------------------------------------------

def write_lammps_script(workdir, file_name, pxy, pxz):
    
    os.makedirs(workdir, exist_ok=True)
    inpath = os.path.join(workdir, "in.lammps")
    out_data = os.path.join(workdir, file_name)

    # BOX
    Lx, Ly, Lz = 96.6666666667, 58.0, 58.0

    # SIMULATION
    numsteps = 5000
    dt = 0.0613 # in ~48*fs 
    bins = 50
    # OBMD
    alpha_obmd = 0.95
    tau_usher = 10.0 # originaly 100.0
    buffer_size = 0.2  # in units of timesteps
    ds0 = 1.0 # originaly 1.0
    dtheta0 = 0.01
    uovlp = 1e4  # za dpd nepomemben
    dsovlp = 5.0  # za dpd nepomemben
    etarget = 6.0 # kemijski potencial (~ -11.0)
    nattempt = 50
    nattempts = 160
    gfac = 0.25
    # STRESS
    pxx = 0.0

    # calculate dependent parameters
    nevery = 100
    buffer_size *= Lx
    shear_size = buffer_size
    gamma_dpd = 100.0
    nbuf = 367

    # write LAMMPS input script
    file1 = f'{workdir}/in.lammps'
    lines1 = open(file1, "w")
    print(pxy)
    text00 = f"""units				lj
    atom_style			full
    dimension			3
    boundary			f p p

    # Set the pair interaction style to Lennard-Jones
    pair_style          hybrid/overlay lj/cut 12.0 coul/cut 12.0 dpd {300.0/503.24} 10.0 12355        # Lennard-Jones potential with a cutoff distance of 10 Å
    bond_style          harmonic           # Harmonic bond style
    angle_style         harmonic           # Harmonic angle style
    dihedral_style      hybrid charmm quadratic           # Harmonic dihedral style
    special_bonds       lj 0.0 0.0 0.0    # exclude interactoins between 1-2, 1-3, and 1-4 atoms

    read_data           {file_name}

    # include other files
    include             ../interactions.lmp          # ubq.lmp contains LJ, bond, angle, and dihedral coefficients
    include             ../exclude_groups.lmp
    # viscosity:
    pair_coeff * * dpd 0 {gamma_dpd} 10.0 # gamma cutoff
    # Coulomb
    pair_coeff * * coul/cut


    neighbor            2.0 bin     # Neighbors within 2.0 Å are binned for efficient calculation, multi instead of bin for dpd
    neigh_modify        every 1 delay 0 check yes

    # Time step for the simulation (in fs)
    timestep            {dt}                # Time step size of 1 femtosecond (3e-12 characteristic time of system)

    # regions -------------------------------------------------------
    region 				leftB block {0.0} {buffer_size} 0.0 {Ly} 0.0 {Lz}
    region 				middleB block {buffer_size} {Lx-buffer_size} 0.0 {Ly} 0.0 {Lz}
    region 				rightB block {Lx-buffer_size} {Lx} 0.0 {Ly} 0.0 {Lz}
    region				leftshear block {buffer_size-shear_size} {buffer_size} 0.0 {Ly} 0.0 {Lz}
    region				rightshear block {Lx-buffer_size} {Lx-buffer_size+shear_size} 0.0 {Ly} 0.0 {Lz}
    region 				leftBa block {0.1*buffer_size} {0.9*buffer_size} {0.0*Ly} {1.0*Ly} {0.0*Lz} {1.0*Lz}
    region 				rightBa block {Lx-0.9*buffer_size} {Lx-0.1*buffer_size} {0.0*Ly} {1.0*Ly} {0.0*Lz} {1.0*Lz}

    variable            shearY equal "{pxy}*(step<{2000})"
    variable            shearZ equal "{pxz}*(step<{2000})"

    group               protein type 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18  #19 20 21 22 23

    fix					1 all nve
    fix 				3 all obmd 19 1 1245 {pxx} v_shearY v_shearZ 0 0 0 0 &
                        region1 leftB region2 rightB region3 leftshear region4 rightshear region5 leftB region6 rightB & 
                        buffersize {buffer_size} shearsize {shear_size} alpha {alpha_obmd} tau {tau_usher} nbuf {nbuf} & 
                        gfac {gfac} step 1 attempt {nattempts} usher 1 {etarget} {ds0} {dtheta0} {uovlp} {dsovlp} 1 {nattempt}                    
    # fix COM at the box center
    fix                 restrainCOM protein spring tether 10.0 {Lx/2} {Ly/2} {Lz/2} 0.0
                        
    # Output settings
    thermo              200
    thermo_style        custom step temp pe ke etotal spcpu

    dump				1 protein custom {nevery} {workdir}/positions.lammpstrj id type x y z vx vy vz fx fy fz

    comm_modify         vel yes
    newton              on
    run					{numsteps}

    write_data			{file_name} nocoeff
    """

    lines1.write(text00)
    lines1.close()
    print(f'LAMMPS input script generated.')

    return inpath, out_data

# shear to direction section ---------------------------------------------------

def run_lammps_in_workdir(workdir, lammps_exe="lmp_mpi", cores_per_env=2):
    cmd = [
        "mpirun",
        "-np", str(cores_per_env),
        lammps_exe,
        "-in", "in.lammps"
    ]
    #"--bind-to", "core", "--map-by", "core",
    
    result = subprocess.run(cmd, cwd=workdir, check=True)
    return result.returncode

def shear_to_positions(pxy, pxz, workdir, lammps_exe=None, nprocs=1, atom_ids=None):

    # prepare paths
    file_name = os.path.join(workdir, os.path.basename(workdir) + ".data")
    #file_name = workdir + '/rotated.data'
    print('filename = ', file_name)
    
    # if the file_name doesn't follow that scheme in your setup, pass file path differently.
    inpath, out_data = write_lammps_script(workdir, file_name, pxy, pxz)

    # run lammps
    if lammps_exe is None:
        lammps_exe='/home/tpotisk/Protein_RL/MD-Basics_new/LAMMPS/lammps_develop/build/lmp_mpi'

    returncode = run_lammps_in_workdir(workdir, lammps_exe=lammps_exe, cores_per_env=8)
    print(f"LAMMPS finished with return code: {returncode}")

    # read data
    positions = read_data(out_data, atom_ids=atom_ids)

    return positions
