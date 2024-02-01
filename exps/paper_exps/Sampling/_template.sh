#!/bin/bash
export TOKENIZERS_PARALLELISM=True

### Required variables
prefix=${prefix:-''}
ckpt_name=${ckpt_name:-''}
input_path=${input_path:-''}
engine=${engine:-''}
batch_size=${batch_size:-''}
max_length=${max_length:-''}
num_return_sequences=${num_return_sequences:-''}
do_sample=${do_sample:-''}

### Default variables
temperature='1.0'
num_processes=8
main_process_port=8889
mode='train'
if [[ "${input_path}" == *"_test_set"* ]]; then
    mode='test'
fi

res_file_name="${ckpt_name}_${num_return_sequences}_temp_${temperature}_do_sample_${do_sample}_${mode}"
model_name="${prefix}/${ckpt_name}"
save_dir="${prefix}/sampling_results/${res_file_name}" 

wandb_log="False"
wandb_project="TMP"
wandb_run_name="${res_file_name}"
accelerate launch --config_file ./default_config.yaml \
    --num_processes=${num_processes} --main_process_port=${main_process_port} \
    sampling.py \
    --model_name "${model_name}" \
    --input_path "${input_path}" \
    --save_dir "${save_dir}" \
    --engine "${engine}" \
    --batch_size "${batch_size}" \
    --max_length "${max_length}" \
    --num_return_sequences "${num_return_sequences}" \
    --temperature "${temperature}" \
    --do_sample "${do_sample}" \
    --wandb_log "${wandb_log}" \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${wandb_run_name}"
