#!/bin/bash
#SBATCH --job-name=gbso
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=512
# #SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --output=rl.out
#SBATCH --error=rl.err

bash clean_all.sh

mkdir -p models model models_cp logs_SAC_protein checkpoints_SAC lammps

eval "$(micromamba shell hook --shell bash)"
micromamba activate rlenv

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self

echo "Starting RL training with SAC and parallel environments"

start=$SECONDS
python3 SAC_prll.py
duration=$(( SECONDS - start ))

# write duration to a file
echo $duration > training_duration.txt
echo "Training completed in $duration seconds"

# create copy of models folder with time in the title
timestamp=$(date +%Y%m%d_%H%M%S)
cp -r models models_cp/models_$timestamp
