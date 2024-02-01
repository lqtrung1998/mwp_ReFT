### gsm8k 
## Python SDP
# Codellama
exp_name="gsm8k_python_sdp_codellama_reft_rerank" \
train_file='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_codellama/sampling_results/global_step_308_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='ppo_paper_final_new/_models_outputs_rl/gsm8k_python_sdp_codellama_reft/sampling_results/best_100_temp_1.0_do_sample_1_test.json' \
model_name_or_path='ppo_paper_final_new/_models_outputs_rl/gsm8k_python_sdp_codellama_reft/best/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_rl/gsm8k_python_sdp_codellama_reft/best/' \
n_epochs='3' \
    bash exps/paper_exps/Rerank/_template.sh

# Galactica
exp_name="gsm8k_python_sdp_galactica_reft_rerank" \
train_file='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_galactica/sampling_results/global_step_308_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='ppo_paper_final_new/_models_outputs_rl/gsm8k_python_sdp_galactica_reft/sampling_results/best_100_temp_1.0_do_sample_1_test.json' \
model_name_or_path='ppo_paper_final_new/_models_outputs_rl/gsm8k_python_sdp_galactica_reft/best/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_rl/gsm8k_python_sdp_galactica_reft/best/' \
n_epochs='3' \
    bash exps/paper_exps/Rerank/_template.sh

## NL
# Codellama
exp_name="gsm8k_nl_codellama_reft_rerank" \
train_file='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_codellama/sampling_results/global_step_312_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='ppo_paper_final_new/_models_outputs_rl/gsm8k_nl_codellama_reft/sampling_results/best_100_temp_1.0_do_sample_1_test.json' \
model_name_or_path='ppo_paper_final_new/_models_outputs_rl/gsm8k_nl_codellama_reft/best/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_rl/gsm8k_nl_codellama_reft/best/' \
n_epochs='3' \
    bash exps/paper_exps/Rerank/_template.sh

# Galactica
exp_name="gsm8k_nl_galactica_reft_rerank" \
train_file='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_galactica/sampling_results/global_step_312_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='ppo_paper_final_new/_models_outputs_rl/gsm8k_nl_galactica_reft/sampling_results/best_100_temp_1.0_do_sample_1_test.json' \
model_name_or_path='ppo_paper_final_new/_models_outputs_rl/gsm8k_nl_galactica_reft/best/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_rl/gsm8k_nl_galactica_reft/best/' \
n_epochs='3' \
    bash exps/paper_exps/Rerank/_template.sh