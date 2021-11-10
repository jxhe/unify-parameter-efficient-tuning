#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=xsum
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=25g
#SBATCH --cpus-per-task=2
#SBATCH --time=0
##SBATCH --array=0


for L in 1 10 30 100 200; do
  bash exps/run_bart_trainer_low_pt.sh $L
done