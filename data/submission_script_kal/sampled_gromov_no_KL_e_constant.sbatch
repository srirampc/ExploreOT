#!/bin/bash
#SBATCH --account=gts-saluru8-coda20
#SBATCH -N1 --ntasks-per-node=8
#SBATCH --time=8:00:00
#SBATCH -qinferno
#SBATCH --output=report_%x_%j.out
#SBATCH --error=report_%x_%j.err
#SBATCH --mail-user=klapborisuth3@gatech.edu

cd $SLURM_SUBMIT_DIR                            

module load python/3.9
module load anaconda3/2021.05
conda activate GW_dist_compare_env

/usr/bin/time -v srun python3 main_comparaison.py -s --n_samples_s $1 --loss_func_name $2 --name_dataset $3 --name_algo sampled_gromov_no_KL_e_constant --pickle_name 2 --batch_size $4 --param1 $4 --iter_epsilon $5 --param2 $5
