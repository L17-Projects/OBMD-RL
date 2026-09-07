#!/bin/bash

rm -r lammps/model_400000/eval* 2>/dev/null
rm model/*.npy 2>/dev/null
rm test.out 2>/dev/null
rm test.err 2>/dev/null

sbatch run_job.sh

