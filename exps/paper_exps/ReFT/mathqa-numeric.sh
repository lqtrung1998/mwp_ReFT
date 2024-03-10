### mathqa-numeric 
## NL
# Codellama
exp_name="mathqa-numeric_nl_codellama_reft" \
train_file='data/mathqa-numeric_nl.json' \
test_file='data/mathqa-numeric_test_set.json' \
engine='nl' \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa-numeric_nl_codellama/global_step_1870_epoch_10/' \
ref_model_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa-numeric_nl_codellama/global_step_1870_epoch_10/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa-numeric_nl_codellama/global_step_1870_epoch_10/' \
n_epochs='300' \
kl_coef='0.01' \
    bash exps/paper_exps/ReFT/_template.sh

# Galactica
exp_name="mathqa-numeric_nl_galactica_reft" \
train_file='data/mathqa-numeric_nl.json' \
test_file='data/mathqa-numeric_test_set.json' \
engine='nl' \
model_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa-numeric_nl_galactica/global_step_1870_epoch_10/' \
ref_model_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa-numeric_nl_galactica/global_step_1870_epoch_10/' \
tokenizer_name_or_path='ppo_paper_final_new/_models_outputs_sft/mathqa-numeric_nl_galactica/global_step_1870_epoch_10/' \
n_epochs='300' \
kl_coef='0.01' \
    bash exps/paper_exps/ReFT/_template.sh

