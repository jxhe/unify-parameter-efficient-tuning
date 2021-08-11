#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=xsum
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=30g
#SBATCH --cpus-per-task=2
#SBATCH --time=0
##SBATCH --array=0

export TRANSFORMERS_CACHE=checkpoints/hf_model
cache_dir=${TRANSFORMERS_CACHE}


# wandb env variables
export WANDB_PROJECT=xsum_effctune
export WANDB_WATCH="false"

DATE=`date +%Y%m%d`
dataset="xsum"

use_prefix="lisa"
exp_name=xsum_bart_${use_prefix}
SAVE=checkpoints/${dataset}/${DATE}/${exp_name}

rm -rf ${SAVE}; mkdir -p ${SAVE}

epochs=30
lr=5e-5
bsz=24
gradient_steps=2
metric=rouge2
ft='none'
# max_eval_examples=1600
logging_steps=100
label_smoothing_factor=0


eval_strategy="epoch"
save_steps=100

python -u examples/pytorch/summarization/run_summarization.py \
    --dataset_name 'xsum' \
    --model_name_or_path 'facebook/bart-large' \
    --cache_dir ${cache_dir} \
    --use_prefix ${use_prefix} \
    --mid_dim 800 \
    --preseqlen 200 \
    --unfreeze_params ${ft} \
    --preprocessing_num_workers 2 \
    --max_source_length 512 \
    --max_target_length 60 \
    --val_max_target_length 60 \
    --num_beams 6 \
    --max_length 60 \
    --min_length 10 \
    --no_repeat_ngram_size 3 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --gradient_accumulation_steps ${gradient_steps} \
    --num_train_epochs ${epochs} \
    --learning_rate ${lr} \
    --fp16 \
    --logging_steps ${logging_steps} \
    --save_total_limit 2 \
    --label_smoothing_factor ${label_smoothing_factor} \
    --evaluation_strategy ${eval_strategy} \
    --save_strategy ${eval_strategy} \
    --save_steps ${save_steps} \
    --eval_steps ${save_steps} \
    --load_best_model_at_end \
    --report_to "wandb" \
    --run_name ${dataset}.${DATE}.${exp_name} \
    --overwrite_output_dir "True" \
    --disable_tqdm "True" \
    --output_dir ${SAVE} 2>&1 | tee ${SAVE}/log.txt
    # --predict_with_generate
    # --metric_for_best_model ${metric} \
    # --greater_is_better "True" \

#rm -rf ${SAVE}/pytorch_model.bin
