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
            if int(lines[i][1]) < 23:
                if write_count == 1:
                    masses.append(0) # correct masses later ----------------------------!!
                position.append([int(lines[i][0]), float(lines[i][2]), float(lines[i][3]), float(lines[i][4])])


    all_positions = np.array(all_positions)
    masses = np.array(masses)

    return all_positions, masses

def read_data(filename):

    type_cutoff = 23  # e.g., 'b23' -> 23
    positions = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the start of the "Atoms # full" section
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms # full"):
            start = i + 2  # skip the header line + empty line
            break

    if start is None:
        raise ValueError("No 'Atoms # full' section found in file.")

    # Read until next empty line
    for line in lines[start:]:
        if line.strip() == "":
            break

        parts = line.split()
        if len(parts) < 7:
            continue  # skip malformed lines

        atom_id = int(parts[0])
        atom_type = int(parts[2])  # LAMMPS "full" format: id, mol, type, q, x, y, z
        x, y, z = map(float, parts[4:7])

        if atom_type < type_cutoff:
            positions.append((atom_id, atom_type, x, y, z))

    # sort by atom id
    positions.sort(key=lambda x: x[0])
    # return only positions
    positions = [(x[2], x[3], x[4]) for x in positions]

    return np.array(positions)

# save lammps file -------------------------------------------------------------

def write_lammps_script(workdir, file_name, pxy, pxz):
    
    os.makedirs(workdir, exist_ok=True)
    inpath = os.path.join(workdir, "in.lammps")
    out_data = os.path.join(workdir, file_name)

    # BOX
    Lx, Ly, Lz = 91, 65, 65

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
    strd = f'{workdir}/ubq_start.data'
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
    dihedral_style      quadratic           # Harmonic dihedral style
    special_bonds       lj 0.0 0.0 0.0    # exclude interactoins between 1-2, 1-3, and 1-4 atoms

    read_data           {file_name}

    # include other files
    include             interaction_no_ions_lj_cut.lmp         # ubq.lmp contains LJ, bond, angle, and dihedral coefficients
    include             exclude_groups.lmp
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

    group               protein type 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 #23

    fix					1 all nve
    fix 				3 all obmd 23 1 1245 {pxx} v_shearY v_shearZ 0 0 0 0 &
                        region1 leftB region2 rightB region3 leftshear region4 rightshear region5 leftB region6 rightB & 
                        buffersize {buffer_size} shearsize {shear_size} alpha {alpha_obmd} tau {tau_usher} nbuf {nbuf} & 
                        gfac {gfac} step 1 attempt {nattempts} usher 1 {etarget} {ds0} {dtheta0} {uovlp} {dsovlp} 1 {nattempt}                    
    # fix COM at the box center
    fix                 restrainCOM protein spring tether 10.0 60.0 36.0 36.0 0.0
                        
    # Output settings
    thermo              200
    thermo_style        custom step temp pe ke etotal spcpu

    #dump				1 all custom {nevery} {workdir}/positions.lammpstrj id type x y z vx vy vz fx fy fz
    #dump_modify         1 append yes # added tp 3.3.2026

    dump				2 protein custom {nevery} {workdir}/protein.lammpstrj id type x y z
    #dump_modify         2 append yes # added tp 3.3.2026


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

def run_lammps_in_workdir_option1(workdir, lammps_exe="lmp_mpi", cores_per_env=4, timeout=None):
    # Use srun to launch one task that consumes cores_per_env cores (from the job allocation)
    # --exclusive ensures the task gets dedicated resources


    cmd = [
        "srun",
        "-n", "1",                         # one SLURM task for this LAMMPS run
        "-c", str(cores_per_env),          # CPUs per task (same as sbatch --cpus-per-task)
        "--exclusive",                     # do not share CPUs with other srun jobs
        "--cpu_bind=cores",                # bind to physical cores (optional; cluster-dependent)
        lammps_exe, "-in", "in.lammps"
    ]
    print('bbb')
    # run in the environment-specific working directory
    result = subprocess.run(cmd, cwd=workdir, check=True)
    print(result)
    return result.returncode

def run_lammps_in_workdir(workdir, lammps_exe="lmp_mpi", cores_per_env=4):
    cmd = [
        "mpirun",
        "-np", str(cores_per_env),
        lammps_exe,
        "-in", "in.lammps"
    ]

    import os 
    n = os.environ.get("SLURM_NTASKS", "1")

    print(f"n tasks  = {n}")

    #import time

    #time.sleep(5)

    #print("Running:", " ".join(cmd))
    prev_cwd = os.getcwd()
    try:
        os.chdir(workdir)

        cmd = f'mpirun -np {cores_per_env} {lammps_exe} -in in.lammps'

        os.system(cmd)
        
        #result = subprocess.run(cmd, cwd=workdir, check=True)
        return 1 #result.returncode
    finally:
        os.chdir(prev_cwd)


def run_lammps_in_workdir_option2(workdir, lammps_exe, cores_per_env=2, timeout=120):
    import subprocess, time, os
    
    cmd = [
        "mpirun", "-np", str(cores_per_env),
        "--bind-to", "core", "--map-by", "core",
        lammps_exe, "-in", "in.lammps"
    ]

    stdout = os.path.join(workdir, "lammps_stdout.txt")
    stderr = os.path.join(workdir, "lammps_stderr.txt")

    with open(stdout, "w") as out, open(stderr, "w") as err:
        try:
            result = subprocess.run(
                cmd,
                cwd=workdir,
                stdout=out,
                stderr=err,
                timeout=timeout,
                check=True
            )
            return result.returncode
        except Exception as e:
            print(f"LAMMPS failed in {workdir}: {e}")
            return -1

def shear_to_positions(pxy, pxz, workdir, lammps_exe=None, nprocs=1):

    # prepare paths
    file_name = os.path.join(workdir, os.path.basename(workdir) + ".data")
    file_name = workdir + '/rotated.data'
    print('filename = ', file_name)
    
    # if the file_name doesn't follow that scheme in your setup, pass file path differently.
    inpath, out_data = write_lammps_script(workdir, file_name, pxy, pxz)

    # run lammps
    if lammps_exe is None:
        lammps_exe='/home/tpotisk/Protein_RL/MD-Basics_new/LAMMPS/lammps_develop/build/lmp_mpi'

    returncode = run_lammps_in_workdir(workdir, lammps_exe=lammps_exe, cores_per_env=8)
    print(f"LAMMPS finished with return code: {returncode}")

    # read data
    positions = read_data(out_data)

    return positions
