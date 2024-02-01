#!/bin/bash

exp_name="svamp_python_sdp_galactica_125m"
config_file="./default_config_deepspeed.yaml"
keep_num_ckpt='40'
batch_size="6"
gradient_accumulation_steps="1"

train_file="data/svamp_python_sdp.json"
test_file="data/svamp_test_set.json"
engine="python" # 'python' or 'nl'
model_name_or_path="hf_models/galactica-125m"
tokenizer_name_or_path="hf_models/galactica-125m"
model_dir="ppo_paper_final_new/_models_outputs_sft_small/${exp_name}/"
wandb_run_name="${exp_name}"
wandb_log="True"
wandb_project="ReFT_small"
n_epochs="40"
num_workers="8"
learning_rate="2e-5"
weight_decay="0"
warmup_step="-100"
clip_grad_norm="1"
evaluating_epoch_freq="1"
logging_epoch_freq="1"
saving_epoch_freq="1"
logging_step_freq="10"
evaluating_step_freq="-100"
saving_step_freq="-100"
seed="42"
max_input_length="1024"

num_processes='8'
main_process_port='8888'

mkdir -p "${model_dir}"
accelerate launch \
            --config_file "${config_file}" \
            --num_processes=${num_processes} \
            --main_process_port=${main_process_port} \
    train_sft_model.py \
            --model_name_or_path "${model_name_or_path}" \
            --tokenizer_name_or_path "${tokenizer_name_or_path}" \
            --train_file "${train_file}" \
            --test_file "${test_file}" \
            --model_dir "${model_dir}" \
            --batch_size "${batch_size}" \
            --n_epochs "${n_epochs}" \
            --num_workers "${num_workers}" \
            --learning_rate "${learning_rate}" \
            --weight_decay "${weight_decay}" \
            --warmup_step "${warmup_step}" \
            --clip_grad_norm "${clip_grad_norm}" \
            --evaluating_epoch_freq "${evaluating_epoch_freq}" \
            --logging_epoch_freq "${logging_epoch_freq}" \
            --saving_epoch_freq "${saving_epoch_freq}" \
            --evaluating_step_freq "${evaluating_step_freq}" \
            --logging_step_freq "${logging_step_freq}" \
            --saving_step_freq "${saving_step_freq}" \
            --seed "${seed}" \
            --max_input_length "${max_input_length}" \
            --gradient_accumulation_steps "${gradient_accumulation_steps}" \
            --keep_num_ckpt "${keep_num_ckpt}" \
            --wandb_log "${wandb_log}" \
            --wandb_project "${wandb_project}" \
            --wandb_run_name "${wandb_run_name}" \
            --engine "${engine}" \
            1> >(tee "${model_dir}"/"${exp_name}".log) \
            2> >(tee "${model_dir}"/"${exp_name}".err >&2)