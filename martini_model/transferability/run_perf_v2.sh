#!/bin/bash
#SBATCH --account=l17
#SBATCH --mem=20GB
#SBATCH --job-name=ubqm
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --array=0-63
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=240:00:00

eval "$(micromamba shell hook --shell bash)"
micromamba activate rlenv

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

EPISODES_PER_JOB=10
START=$(( SLURM_ARRAY_TASK_ID * EPISODES_PER_JOB ))
END=$(( START + EPISODES_PER_JOB ))

shopt -s nullglob
mapfile -t MODEL_FILES < <(printf '%s\n' models/model_*.zip | sort -V)

if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    echo "No models found"
    exit 1
fi

for MODEL_PATH in "${MODEL_FILES[@]}"; do

    MODEL_FILE="$(basename "$MODEL_PATH")"
    MODEL_NAME="${MODEL_FILE%.zip}"

    # 
    MODEL_DIR="lammps/${MODEL_NAME}"
    WORKDIR="${MODEL_DIR}/eval_${SLURM_ARRAY_TASK_ID}"

    mkdir -p "$WORKDIR"

    echo "===================================="
    echo "Model: $MODEL_NAME"
    echo "Task:  $SLURM_ARRAY_TASK_ID"
    echo "Dir:   $WORKDIR"
    echo "===================================="

    cp lammps/ubq_start.data "$WORKDIR/"
	cp lammps/interaction_no_ions_lj_cut.lmp "$WORKDIR/"
	cp lammps/exclude_groups.lmp "$WORKDIR/"

    python algorithm_performance.py \
        "$MODEL_NAME" SAC "$START" "$END" "$WORKDIR"

done