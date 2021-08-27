#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --array=0-0%1
#SBATCH --job-name=xsum
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30g
#SBATCH --cpus-per-task=2
#SBATCH --time=0
##SBATCH --array=0

export TRANSFORMERS_CACHE=checkpoints/hf_model
cache_dir=${TRANSFORMERS_CACHE}


# wandb env variables
export WANDB_PROJECT=xsum_tride
export WANDB_WATCH="false"

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}

DATE=`date +%Y%m%d`
dataset="xsum"

declare -a model_list=("checkpoints/xsum/20210819/xsum_tride.prefix.lisa.cross_attn.ms100000.ls0.1.wd0.01"
    )
arglen=${#model_list[@]}
i=$(( taskid%arglen ))

model_path=${model_list[$i]}
SAVE=${model_path}

log="test_log.txt"

use_prefix="lisa"
lisa_option="cross_attn"
# adapter_option="attn_adapter"
mh_reuse_proj="True"

max_steps=100000
num_train_epochs=30
warmup_updates=0
lr=5e-5
lr_scheduler_type="polynomial"
max_grad_norm=0.1
weight_decay=0.01
bsz=16
gradient_steps=4
metric=rouge2
ft='ef_'
top_layers=12
max_eval_samples=1600
max_train_samples=2000
logging_steps=100
label_smoothing_factor=0.1

eval_strategy="no"
# eval_strategy="steps"
save_steps=3000

extra_cmd=""
debug_str=""

python -u examples/pytorch/summarization/run_summarization.py \
    --dataset_name 'xsum' \
    --model_name_or_path ${model_path} \
    --cache_dir ${cache_dir} \
    --use_prefix ${use_prefix} \
    --lisa_option ${lisa_option} \
    --mh_reuse_proj ${mh_reuse_proj} \
    --mid_dim 800 \
    --preseqlen 200 \
    --init_with_bert 1 \
    --unfreeze_params ${ft} \
    --num_bias_layers ${top_layers} \
    --preprocessing_num_workers 2 \
    --max_source_length 512 \
    --max_target_length 128 \
    --val_max_target_length 60 \
    --max_eval_samples ${max_eval_samples} \
    --num_beams 6 \
    --max_length 60 \
    --min_length 10 \
    --no_repeat_ngram_size 3 \
    --do_predict \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --gradient_accumulation_steps ${gradient_steps} \
    --max_steps ${max_steps} \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${lr} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --max_grad_norm ${max_grad_norm} \
    --weight_decay ${weight_decay} \
    --warmup_steps ${warmup_updates} \
    --fp16 \
    --logging_steps ${logging_steps} \
    --save_total_limit 2 \
    --label_smoothing_factor ${label_smoothing_factor} \
    --evaluation_strategy ${eval_strategy} \
    --save_strategy ${eval_strategy} \
    --save_steps ${save_steps} \
    --eval_steps ${save_steps} \
    --load_best_model_at_end \
    --report_to "none" \
    --run_name ${dataset}.${DATE}.${exp_name} \
    --overwrite_output_dir "False" \
    --disable_tqdm "True" \
    --metric_for_best_model ${metric} \
    --greater_is_better "True" \
    --predict_with_generate \
    --output_dir ${SAVE} ${extra_cmd} \
        2>&1 | tee ${SAVE}/${log}
    # --predict_with_generate
    # --metric_for_best_model ${metric} \
    # --greater_is_better "True" \

#rm -rf ${SAVE}/pytorch_model.bin
