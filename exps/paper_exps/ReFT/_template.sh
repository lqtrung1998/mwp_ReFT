#!/bin/bash
export TOKENIZERS_PARALLELISM=True

### Required variables
exp_name=${exp_name:-''}
train_file=${train_file:-''}
test_file=${test_file:-''}
engine=${engine:-''}
model_name_or_path=${model_name_or_path:-''}
ref_model_name_or_path=${ref_model_name_or_path:-''}
tokenizer_name_or_path=${tokenizer_name_or_path:-''}
n_epochs=${n_epochs:-''}
kl_coef=${kl_coef:-''} # For NL should put 0.05, For Python put 0.01

### Default variables
model_dir="ppo_paper_final_new/_models_outputs_rl/${exp_name}/"
config_file="./default_config_deepspeed.yaml"

batch_size="4"
mini_batch_size="4"
eval_batch_size="8"
ppo_epochs="2"
num_workers="0"
learning_rate="3e-7"
weight_decay="0"
warmup_step="0"
clip_grad_norm="1"
vf_coef="5"
# kl_coef="0.01"
gamma="1.0"
lam="0.95"
adv_whitening='global'
seed="42"
max_input_length="300"
max_gen_length="700"
keep_num_ckpt='0'

evaluating_epoch_freq="1"
logging_epoch_freq="1"
saving_epoch_freq="1"

logging_step_freq="1"
evaluating_step_freq="-100"
saving_step_freq="-100"

wandb_log="True"
wandb_project="ReFT"
wandb_run_name="${exp_name}"
#########

num_processes='8'
main_process_port='8888'

mkdir -p "${model_dir}"
accelerate launch \
            --config_file "${config_file}" \
            --num_processes=${num_processes} \
            --main_process_port=${main_process_port} \
    train_rl_reft.py \
            --model_name_or_path "${model_name_or_path}" \
            --tokenizer_name_or_path "${tokenizer_name_or_path}" \
            --ref_model_name_or_path "${ref_model_name_or_path}" \
            --train_file "${train_file}" \
            --test_file "${test_file}" \
            --model_dir "${model_dir}" \
            --batch_size "${batch_size}" \
            --mini_batch_size "${mini_batch_size}" \
            --eval_batch_size "${eval_batch_size}" \
            --ppo_epochs "${ppo_epochs}" \
            --n_epochs "${n_epochs}" \
            --num_workers "${num_workers}" \
            --learning_rate "${learning_rate}" \
            --weight_decay "${weight_decay}" \
            --warmup_step "${warmup_step}" \
            --clip_grad_norm "${clip_grad_norm}" \
            --vf_coef "${vf_coef}" \
            --kl_coef "${kl_coef}" \
            --gamma "${gamma}" \
            --lam "${lam}" \
            --evaluating_epoch_freq "${evaluating_epoch_freq}" \
            --logging_epoch_freq "${logging_epoch_freq}" \
            --saving_epoch_freq "${saving_epoch_freq}" \
            --evaluating_step_freq "${evaluating_step_freq}" \
            --logging_step_freq "${logging_step_freq}" \
            --saving_step_freq "${saving_step_freq}" \
            --seed "${seed}" \
            --max_input_length "${max_input_length}" \
            --max_gen_length "${max_gen_length}" \
            --wandb_log "${wandb_log}" \
            --wandb_project "${wandb_project}" \
            --wandb_run_name "${wandb_run_name}" \
            --engine "${engine}" \
            --adv_whitening "${adv_whitening}" \
            --keep_num_ckpt "${keep_num_ckpt}" \
            1> >(tee "${model_dir}"/"${exp_name}".log) \
            2> >(tee "${model_dir}"/"${exp_name}".err >&2)
