### GSM8K 
## Python SDP
# Codellama
exp_name="gsm8k_python_sdp_codellama_offsl" \
train_file='data/gsm8k_python_sdp.json | ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_codellama/sampling_results/global_step_308_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='data/gsm8k_test_set.json' \
engine='python' \
model_name_or_path='hf_models/CodeLlama-7b-hf/' \
tokenizer_name_or_path='hf_models/CodeLlama-7b-hf/' \
n_epochs='20' \
self_learning_samples='10' \
logging_epoch_freq='-100' \
evaluating_epoch_freq='-100' \
saving_epoch_freq='-100' \
logging_step_freq='175' \
evaluating_step_freq='175' \
saving_step_freq='175' \
    bash exps/paper_exps/OffSL/_template.sh

# Galactica
exp_name="gsm8k_python_sdp_galactica_offsl" \
train_file='data/gsm8k_python_sdp.json | ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_galactica/sampling_results/global_step_308_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='data/gsm8k_test_set.json' \
engine='python' \
model_name_or_path='hf_models/galactica-6.7b/' \
tokenizer_name_or_path='hf_models/galactica-6.7b/' \
n_epochs='20' \
self_learning_samples='10' \
logging_epoch_freq='-100' \
evaluating_epoch_freq='-100' \
saving_epoch_freq='-100' \
logging_step_freq='175' \
evaluating_step_freq='175' \
saving_step_freq='175' \
    bash exps/paper_exps/OffSL/_template.sh


# ## NL
# # Codellama
exp_name="gsm8k_nl_codellama_offsl" \
train_file='data/gsm8k_nl.json | ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_codellama/sampling_results/global_step_312_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='data/gsm8k_test_set.json' \
engine='nl' \
model_name_or_path='hf_models/CodeLlama-7b-hf/' \
tokenizer_name_or_path='hf_models/CodeLlama-7b-hf/' \
n_epochs='20' \
self_learning_samples='10' \
logging_epoch_freq='-100' \
evaluating_epoch_freq='-100' \
saving_epoch_freq='-100' \
logging_step_freq='175' \
evaluating_step_freq='175' \
saving_step_freq='175' \
    bash exps/paper_exps/OffSL/_template.sh

# Galactica
exp_name="gsm8k_nl_galactica_offsl" \
train_file='data/gsm8k_nl.json | ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_galactica/sampling_results/global_step_312_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='data/gsm8k_test_set.json' \
engine='nl' \
model_name_or_path='hf_models/galactica-6.7b/' \
tokenizer_name_or_path='hf_models/galactica-6.7b/' \
n_epochs='20' \
self_learning_samples='10' \
logging_epoch_freq='-100' \
evaluating_epoch_freq='-100' \
saving_epoch_freq='-100' \
logging_step_freq='175' \
evaluating_step_freq='175' \
saving_step_freq='175' \
    bash exps/paper_exps/OffSL/_template.sh

