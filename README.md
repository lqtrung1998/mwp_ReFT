# ReFT: Reasoning with REinforced Fine-Tuning
This repo contains source code and data to reproduce the results in the research paper [ReFT: Reasoning with REinforced Fine-Tuning](https://arxiv.org/abs/2401.08967)

## Instruction
### SFT
Main script: `train_sft_model.py`

Run SFT:
```
bash exps/paper_exps/SFT/gsm8k.sh # or svamp, mathqa
```
### ReFT
Main script: `train_rl_ppo_v2.py`

Run ReFT:
```
bash exps/paper_exps/ReFT/gsm8k.sh # or svamp, mathqa
```

### Online-SL
Main script: `train_rl_sl_v2.py`

Run Online-SL:
```
bash exps/paper_exps/OnSL/gsm8k.sh # or svamp, mathqa
```

### Offline-SL
Main script: `train_sl_model.py`

First, use one of the checkpoint and run sampling:
```
bash exps/paper_exps/Sampling/gsm8k.sh # or svamp, mathqa
```

Then configure the train data path and run Offline-SL:
```
bash exps/paper_exps/OffSL/gsm8k.sh # or svamp, mathqa
```

### Top-1 and voting Acc
Main script: `sampling.py`

Configure variables e.g num_return_sequences=100, do_sample=1.0 for voting@100, then run sampling:
```
bash exps/paper_exps/Sampling/gsm8k.sh # or svamp, mathqa
```

### Reranking
Main script: `train_reward_model.py`

First, use one of the (earlier) checkpoint to run sampling on the train set and use the best checkpoint to run sampling on the test set.
```
bash exps/paper_exps/Sampling/gsm8k.sh 
bash exps/paper_exps/Sampling/gsm8k_test.sh # or gsm8k_reft_test
```
Configure the data path and train the rerank model:
```
bash exps/paper_exps/Rerank/gsm8k.sh # or gsm8k_reft
```

## Checkpoints
We provide checkpoints for Codellama model after Reinforced Finetuned on GSM8k and the corresponding Rerank model at: [Codellama-7b-hf-ReFT-GSM8k](https://huggingface.co/lqtrung1998/Codellama-7b-hf-ReFT-GSM8k) and [Codellama-7b-hf-ReFT-Rerank-GSM8k](https://huggingface.co/lqtrung1998/Codellama-7b-hf-ReFT-Rerank-GSM8k)

Note: Our models are tuned based on Codellama, thus, licenses applicable to Codellama, such as [Llama license](https://github.com/lqtrung1998/mwp_ReFT/blob/main/Llama_License.txt), also hold on these models

## License:
[Apache2.0 License](https://github.com/lqtrung1998/mwp_ReFT/blob/main/License.txt)

## Citation
Please cite the paper if you use our data, model or code.
```
@misc{luong2024reft,
      title={ReFT: Reasoning with Reinforced Fine-Tuning}, 
      author={Trung Quoc Luong and Xinbo Zhang and Zhanming Jie and Peng Sun and Xiaoran Jin and Hang Li},
      year={2024},
      eprint={2401.08967},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
