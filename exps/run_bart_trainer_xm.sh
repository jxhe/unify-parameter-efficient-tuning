#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=6.MAM:PT30.PA512
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=gpu
#SBATCH --mem=40g
#SBATCH --cpus-per-task=3
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1

seeds=(15217 65537)
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

source activate iclr
which python

export TRANSFORMERS_CACHE=pretrain_models/huggingface
export HF_DATASETS_CACHE=pretrain_models/huggingface
export HF_METRICS_CACHE=pretrain_models/huggingface
cache_dir=pretrain_models/huggingface

export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# wandb env variables
export WANDB_PROJECT=xsum_tride
export WANDB_WATCH="false"

DATE=`date +%Y%m%d`
dataset="xsum"

# placeholder arguments
adapter_init_option="bert"
adapter_layernorm_option="none"
adapter_scalar=0
lora_alpha=1
lora_init="lora"
lora_dropout=0.1
attn_gate="none"
ffn_gate="none"

# 1.pfeiffer.adapter.600
attn_mode="none"
attn_option="none"
ffn_mode="adapter"
ffn_option="pfeiffer"
preseqlen=1
ffn_bn_len=600
adapter_init_option="bert"
adapter_layernorm_option="none"
adapter_scalar=1

# 2.lora.ffn.102
attn_mode="none"
attn_option="none"
ffn_mode="lora"
ffn_option="none"
preseqlen=1
ffn_bn_len=102
lora_alpha=408
lora_init="lora"
lora_dropout=0.1

# 3.PA.1024
#attn_mode="none"
#attn_option="none"
#ffn_mode="adapter"
#ffn_option="ffn_hi_input"
#preseqlen=1
#ffn_bn_len=1024
#adapter_init_option="bert"
#adapter_layernorm_option="none"
#adapter_scalar=1

# 4.PA30.PA512
#attn_mode="adapter"
#attn_option="attn_adapter"
#ffn_mode="adapter"
#ffn_option="ffn_hi_input"
#preseqlen=30
#ffn_bn_len=512
#adapter_init_option="bert"
#adapter_layernorm_option="none"
#adapter_scalar=1

# 5.PT30.LoRA102
#attn_mode="lisa"
#attn_option="concat"
#ffn_mode="lora"
#ffn_option="none"
#preseqlen=30
#ffn_bn_len=102
#lora_alpha=204
#lora_init="lora"
#lora_dropout=0.1

# 6.MAM:PT30.PA512
attn_mode="lisa"
attn_option="concat"
ffn_mode="adapter"
ffn_option="ffn_hi_input"
preseqlen=30
ffn_bn_len=512
adapter_init_option="lora"
adapter_layernorm_option="fixed_scalar"
adapter_scalar=4

mh_reuse_proj="True"
weight_decay=0.01
max_steps=95000
num_train_epochs=30
warmup_updates=0
lr=5e-5
lr_scheduler_type="polynomial"
max_grad_norm=0.1
bsz=16
gradient_steps=4
metric=rouge2
ft='ef_'
top_layers=12
max_eval_samples=1600
max_train_samples=2000
logging_steps=100
label_smoothing_factor=0.1

eval_strategy="steps"
save_steps=3000
report_to="wandb"

debug=0
extra_cmd=""
debug_str=""

if [ "${debug}" = 1 ];
then
    label_smoothing_factor=0
    weight_decay=0
    max_grad_norm=1
    max_train_samples=2000
    bsz=24
    gradient_steps=2
    num_train_epochs=30
    max_steps=-1
    eval_strategy='steps'
    save_steps=100
    report_to="none"
    logging_steps=10
    extra_cmd="--max_train_samples ${max_train_samples}"
    debug_str=".debug"
fi

exp_name=xsum_tride.am_${attn_mode}.ao_${attn_option}.fm_${ffn_mode}.fo_${ffn_option}.abn${preseqlen}.fbn${ffn_bn_len}.ainit_${adapter_init_option}.alo_${adapter_layernorm_option}.as_${adapter_scalar}.lora_alpha_dropout_${lora_alpha}_${lora_dropout}.lorainit_${lora_init}.unfreeze_${ft}.ms${max_steps}.ls${label_smoothing_factor}.warm${warmup_updates}.wd${weight_decay}.seed${SEED}.${debug_str}
SAVE=checkpoints/${dataset}/${DATE}/${exp_name}
rm -rf ${SAVE}; mkdir -p ${SAVE}
rm ${HF_DATASETS_CACHE}/downloads/*.lock
rm ${HF_DATASETS_CACHE}/*.lock
cp ${0} ${SAVE}/run.sh

python -u examples/pytorch/summarization/run_summarization.py \
    --dataset_name 'xsum' \
    --model_name_or_path 'facebook/bart-large' \
    --cache_dir ${cache_dir} \
    --seed ${SEED} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --lora_init ${lora_init} \
    --attn_mode ${attn_mode} \
    --attn_option ${attn_option} \
    --ffn_mode ${ffn_mode} \
    --ffn_option ${ffn_option} \
    --attn_gate ${attn_gate} \
    --ffn_gate ${ffn_gate} \
    --adapter_layernorm_option ${adapter_layernorm_option} \
    --adapter_init_option ${adapter_init_option} \
    --adapter_scalar ${adapter_scalar} \
    --mh_reuse_proj ${mh_reuse_proj} \
    --mid_dim 800 \
    --preseqlen ${preseqlen} \
    --ffn_bn_len ${ffn_bn_len} \
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
    --do_train \
    --do_eval \
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
    --report_to ${report_to} \
    --run_name ${dataset}.${DATE}.${exp_name} \
    --overwrite_output_dir "True" \
    --disable_tqdm "True" \
    --metric_for_best_model ${metric} \
    --greater_is_better "True" \
    --predict_with_generate \
    --output_dir ${SAVE} ${extra_cmd} 2>&1 | tee ${SAVE}/log.txt
