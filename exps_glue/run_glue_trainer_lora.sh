#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=glue
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=30g
#SBATCH --cpus-per-task=3
#SBATCH --time=0
##SBATCH --array=0

export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model

cache_dir=${TRANSFORMERS_CACHE}


# TASK_NAME=sst2
metric="accuracy"
TASK_NAME=mnli
# wandb env variables
export WANDB_PROJECT=glue.${TASK_NAME}
export WANDB_WATCH="false"

DATE=`date +%Y%m%d`

# declare -a seed_list=(42)
declare -a seed_list=(42 2 4 6 8)

port=62221
# Hi adapter200
attn_mode="lora"
attn_option="none"
ffn_mode="none"
ffn_option="none"
preseqlen=16
lora_alpha=32
ffn_bn_len=16
lora_dropout=0.1
hi_lnbefore=1
adapter_layernorm_option="none"
max_grad_norm=1
attn_gate="none"
ffn_gate="none"

debug=0
report_to="wandb"

bsz=32
gradient_steps=1

lr=1e-4
# lr=2e-5
weight_decay=0.1
warmup_updates=0
warmup_ratio=0.06
max_steps=-1
num_train_epochs=10
max_tokens_per_batch=0
max_seq_length=512

layer_norm_in=1
layer_norm_out=0
mh_reuse_proj="True"
lr_scheduler_type="polynomial"
#metric=bleu
ft='ef_'
top_layers=12
max_eval_samples=1600
logging_steps=50

eval_strategy="epoch"
save_steps=5000

extra_cmd=""
debug_str=""

if [ "${debug}" = 1 ];
then
    weight_decay=0
    max_grad_norm=1
    max_train_samples=1000
    max_eval_samples=150
    bsz=10
    gradient_steps=1
    num_train_epochs=5
    max_steps=-1
    eval_strategy='steps'
    save_steps=100
    report_to="none"
    logging_steps=10
    extra_cmd="--max_train_samples ${max_train_samples} --max_predict_samples 150"
    debug_str=".debug"
fi


for seed in "${seed_list[@]}"; do

    exp_name=glue.${TASK_NAME}.am_${attn_mode}.ao_${attn_option}.fm_${ffn_mode}.fo_${ffn_option}.abn${preseqlen}.fbn${ffn_bn_len}.lora_alpha_dropout_${lora_alpha}_${lora_dropout}.ag_${attn_gate}.fg_${ffn_gate}.adalo_${adapter_layernorm_option}.hilnb_${hi_lnbefore}.uf_${ft}.ne${num_train_epochs}.warm${warmup_ratio}.wd${weight_decay}.seed${seed}.${debug_str}
    SAVE=checkpoints/glue/${TASK_NAME}/${DATE}/${exp_name}
    rm -rf ${SAVE}; mkdir -p ${SAVE}

    rm checkpoints/hf_model/downloads/*.lock
    rm checkpoints/hf_model/*.lock


    # python -m torch.distributed.launch --nproc_per_node 2 --master_port=${port} examples/pytorch/text-classification/run_glue.py \

    python -u examples/pytorch/text-classification/run_glue.py \
        --model_name_or_path roberta-base \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size ${bsz} \
        --per_device_eval_batch_size ${bsz} \
        --max_tokens_per_batch ${max_tokens_per_batch} \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
        --attn_mode ${attn_mode} \
        --attn_option ${attn_option} \
        --attn_gate ${attn_gate} \
        --ffn_mode ${ffn_mode} \
        --ffn_option ${ffn_option} \
        --ffn_gate ${ffn_gate} \
        --lora_alpha ${lora_alpha} \
        --lora_dropout ${lora_dropout} \
        --adapter_layernorm_option ${adapter_layernorm_option} \
        --mh_reuse_proj ${mh_reuse_proj} \
        --layer_norm_before ${layer_norm_in} \
        --layer_norm_after ${layer_norm_out} \
        --hi_lnbefore ${hi_lnbefore} \
        --mid_dim 800 \
        --preseqlen ${preseqlen} \
        --ffn_bn_len ${ffn_bn_len} \
        --init_with_bert 1 \
        --seed ${seed} \
        --unfreeze_params ${ft} \
        --num_bias_layers ${top_layers} \
        --max_eval_samples ${max_eval_samples} \
        --gradient_accumulation_steps ${gradient_steps} \
        --max_steps ${max_steps} \
        --num_train_epochs ${num_train_epochs} \
        --learning_rate ${lr} \
        --lr_scheduler_type ${lr_scheduler_type} \
        --max_grad_norm ${max_grad_norm} \
        --weight_decay ${weight_decay} \
        --warmup_steps ${warmup_updates} \
        --warmup_ratio ${warmup_ratio} \
        --max_seq_length ${max_seq_length} \
        --fp16 \
        --logging_steps ${logging_steps} \
        --save_total_limit 2 \
        --evaluation_strategy ${eval_strategy} \
        --save_strategy ${eval_strategy} \
        --save_steps ${save_steps} \
        --eval_steps ${save_steps} \
        --load_best_model_at_end \
        --report_to ${report_to} \
        --run_name ${TASK_NAME}.${DATE}.${exp_name} \
        --overwrite_output_dir \
        --disable_tqdm "True" \
        --metric_for_best_model ${metric} \
        --greater_is_better "True" \
        --ddp_find_unused_parameter "False" \
        --output_dir ${SAVE} ${extra_cmd} \
            2>&1 | tee ${SAVE}/log.txt
done

