#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=xsum
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=30g
#SBATCH --cpus-per-task=2
#SBATCH -w tir-0-28
#SBATCH --time=0
##SBATCH --array=0

source activate nmt
export TRANSFORMERS_CACHE=checkpoints/hf_model
cache_dir=${TRANSFORMERS_CACHE}

DATE=`date +%Y%m%d`
dataset="xsum"

exp_name=xsum_test
SAVE=checkpoints/${dataset}/${exp_name}
mkdir -p ${SAVE}

model_path=/home/chuntinz/tir5/tride/checkpoints/xsum

epochs=30
lr=5e-5
bsz=24
gradient_steps=2
metric=rouge2
ft='LN+PE'
eval_batch=150
use_prefix="none"

python -u examples/pytorch/summarization/run_summarization_no_trainer.py \
    --dataset_name 'xsum' \
    --model_name_or_path ${model_path} \
    --cache_dir ${cache_dir} \
    --max_val_batches ${eval_batch} \
    --use_prefix ${use_prefix} \
    --mid_dim 800 \
    --preseqlen 200 \
    --preprocessing_num_workers 2 \
    --max_source_length 512 \
    --max_target_length 60 \
    --val_max_target_length 60 \
    --test_max_target_length 100 \
    --eval_max_length 62 \
    --eval_min_length 11 \
    --num_beam 6 \
    --do_predict \
    --val_metric ${metric} \
    --per_device_train_batch_size ${bsz} \
    --gradient_accumulation_steps ${gradient_steps} \
    --num_train_epochs ${epochs} \
    --learning_rate ${lr} \
    --fp16 \
    --output_dir ${SAVE} 2>&1 | tee ${SAVE}/log.txt

#rm -rf ${SAVE}/pytorch_model.bin