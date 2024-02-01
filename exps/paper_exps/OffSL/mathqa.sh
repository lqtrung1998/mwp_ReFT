### mathqa 
## Python SDP
# Codellama
exp_name="mathqa_python_sdp_codellama_offsl" \
train_file='data/mathqa_python_sdp.json | ppo_paper_final_new/_models_outputs_sft/mathqa_python_sdp_codellama/sampling_results/global_step_636_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='data/mathqa_test_set.json' \
engine='python' \
model_name_or_path='hf_models/CodeLlama-7b-hf/' \
tokenizer_name_or_path='hf_models/CodeLlama-7b-hf/' \
n_epochs='20' \
self_learning_samples='10' \
logging_epoch_freq='-100' \
evaluating_epoch_freq='-100' \
saving_epoch_freq='-100' \
logging_step_freq='350' \
evaluating_step_freq='350' \
saving_step_freq='350' \
    bash exps/paper_exps/OffSL/_template.sh

# Galactica
exp_name="mathqa_python_sdp_galactica_offsl" \
train_file='data/mathqa_python_sdp.json | ppo_paper_final_new/_models_outputs_sft/mathqa_python_sdp_galactica/sampling_results/global_step_636_epoch_2_100_temp_1.0_do_sample_1_train.json' \
test_file='data/mathqa_test_set.json' \
engine='python' \
model_name_or_path='hf_models/galactica-6.7b/' \
tokenizer_name_or_path='hf_models/galactica-6.7b/' \
n_epochs='20' \
self_learning_samples='10' \
logging_epoch_freq='-100' \
evaluating_epoch_freq='-100' \
saving_epoch_freq='-100' \
logging_step_freq='350' \
evaluating_step_freq='350' \
saving_step_freq='350' \
    bash exps/paper_exps/OffSL/_template.sh


# ## NL
# # Codellama
exp_name="mathqa_nl_codellama_offsl" \
train_file='data/mathqa_nl.json | ppo_paper_final_new/_models_outputs_sft/mathqa_nl_codellama/sampling_results/global_step_1550_epoch_5_100_temp_1.0_do_sample_1_train.json' \
test_file='data/mathqa_test_set.json' \
engine='nl' \
model_name_or_path='hf_models/CodeLlama-7b-hf/' \
tokenizer_name_or_path='hf_models/CodeLlama-7b-hf/' \
n_epochs='20' \
self_learning_samples='10' \
logging_epoch_freq='-100' \
evaluating_epoch_freq='-100' \
saving_epoch_freq='-100' \
logging_step_freq='350' \
evaluating_step_freq='350' \
saving_step_freq='350' \
    bash exps/paper_exps/OffSL/_template.sh

# Galactica
exp_name="mathqa_nl_galactica_offsl" \
train_file='data/mathqa_nl.json | ppo_paper_final_new/_models_outputs_sft/mathqa_nl_galactica/sampling_results/global_step_1550_epoch_5_100_temp_1.0_do_sample_1_train.json' \
test_file='data/mathqa_test_set.json' \
engine='nl' \
model_name_or_path='hf_models/galactica-6.7b/' \
tokenizer_name_or_path='hf_models/galactica-6.7b/' \
n_epochs='20' \
self_learning_samples='10' \
logging_epoch_freq='-100' \
evaluating_epoch_freq='-100' \
saving_epoch_freq='-100' \
logging_step_freq='350' \
evaluating_step_freq='350' \
saving_step_freq='350' \
    bash exps/paper_exps/OffSL/_template.sh

