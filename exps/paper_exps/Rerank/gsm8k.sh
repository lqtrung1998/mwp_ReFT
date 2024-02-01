### gsm8k 
## Python SDP
# Codellama
exp_name="gsm8k_python_sdp_codellama_rerank" \
train_file='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_codellama/sampling_results/global_step_308_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_codellama/sampling_results/global_step_2926_epoch_19_100_temp_1.0_do_sample_1_test.json' \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_codellama/global_step_2926_epoch_19/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_codellama/global_step_2926_epoch_19/' \
n_epochs='3' \
    bash exps/paper_exps/Rerank/_template.sh

# Galactica
exp_name="gsm8k_python_sdp_galactica_rerank" \
train_file='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_galactica/sampling_results/global_step_308_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_galactica/sampling_results/global_step_6160_epoch_40_100_temp_1.0_do_sample_1_test.json' \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_galactica/global_step_6160_epoch_40/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_galactica/global_step_6160_epoch_40/' \
n_epochs='3' \
    bash exps/paper_exps/Rerank/_template.sh

## NL
# Codellama
exp_name="gsm8k_nl_codellama_rerank" \
train_file='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_codellama/sampling_results/global_step_312_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_codellama/sampling_results/global_step_4524_epoch_29_100_temp_1.0_do_sample_1_test.json' \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_codellama/global_step_4524_epoch_29/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_codellama/global_step_4524_epoch_29/' \
n_epochs='3' \
    bash exps/paper_exps/Rerank/_template.sh

# Galactica
exp_name="gsm8k_nl_galactica_rerank" \
train_file='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_galactica/sampling_results/global_step_312_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_galactica/sampling_results/global_step_5304_epoch_34_100_temp_1.0_do_sample_1_test.json' \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_galactica/global_step_5304_epoch_34/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_galactica/global_step_5304_epoch_34/' \
n_epochs='3' \
    bash exps/paper_exps/Rerank/_template.sh