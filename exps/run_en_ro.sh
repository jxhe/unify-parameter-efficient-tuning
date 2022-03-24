#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=tran
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=30g
#SBATCH --cpus-per-task=3
#SBATCH --time=0
##SBATCH --array=0

export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model

cache_dir=${TRANSFORMERS_CACHE}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo ${SCRIPT_DIR}

# wandb env variables
# export WANDB_PROJECT=enro_translation
# export WANDB_WATCH="false"

DATE=`date +%Y%m%d`
dataset="wmt16"

# ----- MAM adapter -----
attn_mode="prefix"
attn_option="concat"
attn_composition="add"
attn_bn=30  # attn bottleneck dim

ffn_mode="adapter"
ffn_option="parallel"
ffn_adapter_layernorm_option="none"
ffn_adapter_init_option="lora"
ffn_adapter_scalar="4"
ffn_bn=512 # ffn bottleneck dim


# lora params are not set
if [ -z ${lora_alpha+x} ];
then
    lora_alpha=0
    lora_init="lora"
    lora_dropout=0
fi

# set to 1 for debug mode which only
# uses 1600 training examples
debug=0

# set to "wandb" to use weights & bias
report_to="none"

label_smoothing_factor=0.1
weight_decay=0.01

# the prefix tuning baseline prefers the
# commented hyperparam
# label_smoothing_factor=0
# weight_decay=0

# note that the bsz argument is only effective at evaluation but
# does not influence the training -- it is overridden by
# max_tokens_per_batch
bsz=10
max_steps=50000
max_tokens_per_batch=4096
gradient_steps=4

num_train_epochs=30
warmup_updates=0
max_grad_norm=1
lr=5e-5
lr_scheduler_type="polynomial"
#metric=bleu
metric=loss
unfreeze='ef_'
max_eval_samples=1600
logging_steps=100

eval_strategy="steps"
save_steps=5000

extra_cmd=""
debug_str=""

if [ "${debug}" = 1 ];
then
    label_smoothing_factor=0
    weight_decay=0
    max_grad_norm=1
    max_train_samples=4000
    max_eval_samples=150
    bsz=10
    gradient_steps=4
    num_train_epochs=30
    max_steps=-1
    eval_strategy='steps'
    save_steps=100
    report_to="none"
    logging_steps=10
    extra_cmd="--max_train_samples ${max_train_samples} --max_predict_samples 150"
    debug_str=".debug"
fi

exp_name=wmt16_roen.am_${attn_mode}.ao_${attn_option}.fm_${ffn_mode}
exp_name+=.fo_${ffn_option}.abn${preseqlen}.fbn${ffn_bn_len}.ac_${attn_composition}
exp_name+=.fl_${ffn_adapter_layernorm_option}.finit_${ffn_adapter_init_option}
exp_name+=.fs_${ffn_adapter_scalar}.unfrz_${unfreeze}
exp_name+=.ms${max_steps}.ls${label_smoothing_factor}.warm${warmup_updates}
exp_name+=.wd${weight_decay}.mt${max_tokens_per_batch}.${debug_str}
SAVE=checkpoints/${dataset}/${DATE}/${exp_name}
rm -rf ${SAVE}; mkdir -p ${SAVE}

rm checkpoints/hf_model/downloads/*.lock
rm checkpoints/hf_model/*.lock

# python -m torch.distributed.launch --nproc_per_node 2 --master_port=${port} examples/pytorch/translation/run_translation.py \
python -u examples/pytorch/translation/run_translation.py \
    --dataset_name ${dataset}\
    --dataset_config_name ro-en \
    --model_name_or_path "facebook/mbart-large-cc25" \
    --cache_dir ${cache_dir} \
    --source_lang en_XX \
    --target_lang ro_RO \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --max_tokens_per_batch ${max_tokens_per_batch} \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --dropout 0.1 \
    --attention_dropout 0.0 \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --lora_init ${lora_init} \
    --attn_mode ${attn_mode} \
    --attn_option ${attn_option} \
    --attn_composition ${attn_composition} \
    --ffn_mode ${ffn_mode} \
    --ffn_option ${ffn_option} \
    --ffn_adapter_layernorm_option ${ffn_adapter_layernorm_option} \
    --ffn_adapter_scalar ${ffn_adapter_scalar} \
    --ffn_adapter_init_option ${ffn_adapter_init_option} \
    --mid_dim 800 \
    --attn_bn ${attn_bn} \
    --ffn_bn ${ffn_bn} \
    --unfreeze_params ${unfreeze} \
    --preprocessing_num_workers 2 \
    --max_source_length 150 \
    --max_target_length 150 \
    --val_max_target_length 150 \
    --max_eval_samples ${max_eval_samples} \
    --num_beams 5 \
    --max_length 200 \
    --min_length 1 \
    --no_repeat_ngram_size 0 \
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
    --overwrite_output_dir \
    --disable_tqdm "True" \
    --metric_for_best_model ${metric} \
    --greater_is_better "False" \
    --ddp_find_unused_parameter "False" \
    --predict_with_generate \
    --output_dir ${SAVE} ${extra_cmd} 2>&1 | tee ${SAVE}/log.txt

# cd ${SAVE}
bash exps/romanian_postprocess.sh ${SAVE}/test_generated_predictions.txt ${SAVE}/test_gold_labels.txt | tee -a ${SAVE}/log.txt
