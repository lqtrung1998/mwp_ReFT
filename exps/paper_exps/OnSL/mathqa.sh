### mathqa 
## Python SDP
# Codellama
exp_name="mathqa_python_sdp_codellama_onsl" \
train_file="data/mathqa_python_sdp.json" \
test_file="data/mathqa_test_set.json" \
engine="python" \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa_python_sdp_codellama/global_step_636_epoch_2/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa_python_sdp_codellama/global_step_636_epoch_2/' \
n_epochs='300' \
    bash exps/paper_exps/OnSL/_template.sh

# Galactica
exp_name="mathqa_python_sdp_galactica_onsl" \
train_file="data/mathqa_python_sdp.json" \
test_file="data/mathqa_test_set.json" \
engine="python" \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa_python_sdp_galactica/global_step_636_epoch_2/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa_python_sdp_galactica/global_step_636_epoch_2/' \
n_epochs='300' \
    bash exps/paper_exps/OnSL/_template.sh


## NL
# Codellama
exp_name="mathqa_nl_codellama_onsl" \
train_file="data/mathqa_nl.json" \
test_file="data/mathqa_test_set.json" \
engine="nl" \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa_nl_codellama/global_step_1550_epoch_5/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa_nl_codellama/global_step_1550_epoch_5/' \
n_epochs='300' \
    bash exps/paper_exps/OnSL/_template.sh

# Galactica
exp_name="mathqa_nl_galactica_onsl" \
train_file="data/mathqa_nl.json" \
test_file="data/mathqa_test_set.json" \
engine="nl" \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa_nl_galactica/global_step_1550_epoch_5/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa_nl_galactica/global_step_1550_epoch_5/' \
n_epochs='300' \
    bash exps/paper_exps/OnSL/_template.sh
