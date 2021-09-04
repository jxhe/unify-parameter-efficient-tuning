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

export TRANSFORMERS_CACHE=checkpoints/hf_model
cache_dir=${TRANSFORMERS_CACHE}


# wandb env variables
export WANDB_PROJECT=xsum_tride
export WANDB_WATCH="false"

DATE=`date +%Y%m%d`
dataset="xsum"

attn_mode="none"
attn_option="cross_attn"

ffn_mode="none"
ffn_option="ffn_hi_input"

gate_option="none"

layer_norm_in=1
layer_norm_out=0

preseqlen=1
ffn_bn_len=1

# adapter_option="attn_adapter"
mh_reuse_proj="True"

max_steps=100000
num_train_epochs=30
warmup_updates=0
lr=3e-5
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

eval_strategy="steps"
# eval_strategy="steps"
save_steps=3000
report_to="wandb"

debug=0
vis_analysis=0
extra_cmd=""
debug_str=""
analysis_opt=""

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


exp_name=xsum_tride.am_${attn_mode}.ao_${attn_option}.fm_${ffn_mode}.fo_${ffn_option}.go_${gate_option}.abn${preseqlen}.fbn${ffn_bn_len}.lni${layer_norm_in}.lno${layer_norm_out}.unfreeze_${ft}.ms${max_steps}.ls${label_smoothing_factor}.warm${warmup_updates}.wd${weight_decay}${debug_str}
SAVE=checkpoints/${dataset}/${DATE}/${exp_name}

rm -rf ${SAVE}; mkdir -p ${SAVE}

if [ "${vis_analysis}=1" ];
then
    max_train_samples=2000
    num_train_epochs=10
    analysis_opt=${SAVE}/analysis_opt
    mkdir -p ${analysis_opt}
fi

python -u examples/pytorch/summarization/run_summarization.py \
    --dataset_name 'xsum' \
    --model_name_or_path 'facebook/bart-large' \
    --cache_dir ${cache_dir} \
    --attn_mode ${attn_mode} \
    --attn_option ${attn_option} \
    --ffn_mode ${ffn_mode} \
    --ffn_option ${ffn_option} \
    --gate_option ${gate_option} \
    --mh_reuse_proj ${mh_reuse_proj} \
    --layer_norm_before ${layer_norm_in} \
    --layer_norm_after ${layer_norm_out} \
    --analysis_opt ${analysis_opt} \
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
    --output_dir ${SAVE} ${extra_cmd} \
        2>&1 | tee ${SAVE}/log.txt
    # --predict_with_generate
    # --metric_for_best_model ${metric} \
    # --greater_is_better "True" \

#rm -rf ${SAVE}/pytorch_model.bin
