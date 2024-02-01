### mathqa 
# Python SDP
# Codellama
exp_name="mathqa_python_sdp_codellama" \
train_file='data/mathqa_python_sdp.json' \
test_file='data/mathqa_test_set.json' \
engine='python' \
model_name_or_path='hf_models/CodeLlama-7b-hf' \
tokenizer_name_or_path='hf_models/CodeLlama-7b-hf/' \
n_epochs='40' \
    bash exps/paper_exps/SFT/_template.sh

# Galactica
exp_name="mathqa_python_sdp_galactica" \
train_file='data/mathqa_python_sdp.json' \
test_file='data/mathqa_test_set.json' \
engine='python' \
model_name_or_path='hf_models/galactica-6.7b/' \
tokenizer_name_or_path='hf_models/galactica-6.7b/' \
n_epochs='40' \
    bash exps/paper_exps/SFT/_template.sh

# NL
# Codellama
exp_name="mathqa_nl_codellama" \
train_file='data/mathqa_nl.json' \
test_file='data/mathqa_test_set.json' \
engine='nl' \
model_name_or_path='hf_models/CodeLlama-7b-hf' \
tokenizer_name_or_path='hf_models/CodeLlama-7b-hf/' \
n_epochs='40' \
    bash exps/paper_exps/SFT/_template.sh

# Galactica
exp_name="mathqa_nl_galactica" \
train_file='data/mathqa_nl.json' \
test_file='data/mathqa_test_set.json' \
engine='nl' \
model_name_or_path='hf_models/galactica-6.7b/' \
tokenizer_name_or_path='hf_models/galactica-6.7b/' \
n_epochs='40' \
    bash exps/paper_exps/SFT/_template.sh
