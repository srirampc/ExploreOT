#!/bin/bash
#SBATCH --account=gts-saluru8-coda20
#SBATCH -N1 --ntasks-per-node=8
#SBATCH --time=16:00:00
#SBATCH -qinferno
#SBATCH --output=./Report/sliced_gromov/%x.out
#SBATCH --error=./Report/sliced_gromov/%x.err
#SBATCH --mail-user=klapborisuth3@gatech.edu

cd $SLURM_SUBMIT_DIR                            
cd ..

module load python/3.9
module load anaconda3/2021.05
conda activate GW_dist_compare_env

/usr/bin/time -v srun python3 main_comparaison.py -s --n_samples_s $1 --loss_func_name $2 --name_dataset $3 --name_algo sliced_gromov --pickle_name 2
