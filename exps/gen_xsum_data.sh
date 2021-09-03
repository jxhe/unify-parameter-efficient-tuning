#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=xsum
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=30g
#SBATCH --cpus-per-task=2
#SBATCH --time=0
##SBATCH --array=0

source activate tride
which python

export TRANSFORMERS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
cache_dir=/home/chuntinz/tir5/pretrain_models/huggingface

N=1000
output_dir=/projects/tir5/users/chuntinz/data/tride/xsum/xsum_1k
mkdir -p ${output_dir}
DATE=`date +%Y%m%d`

max_eval_samples=1600
max_train_samples=2000

python -u examples/pytorch/summarization/run_summarization_dataset.py \
    --dataset_name 'xsum' \
    --model_name_or_path 'facebook/bart-large' \
    --cache_dir ${cache_dir} \
    --preprocessing_num_workers 2 \
    --num_train_lines ${N} \
    --output_dir ${output_dir}
