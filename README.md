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
Main script: `train_rl_reft.py`

Run ReFT:
```
bash exps/paper_exps/ReFT/gsm8k.sh # or svamp, mathqa
```

### Online-SL
Main script: `train_rl_sl.py`

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
We provide checkpoints for some Galactica and Codellama models at different stages: warmup-SFT, SFT, SFT-Rerank, ReFT and ReFT-Rerank
* [Codellama-7b-hf-SFT-warmup-GSM8k](https://huggingface.co/lqtrung1998/Codellama-7b-hf-SFT-warmup-GSM8k)
* [Codellama-7b-hf-SFT-GSM8k](https://huggingface.co/lqtrung1998/Codellama-7b-hf-SFT-GSM8k)
* [Codellama-7b-hf-SFT-Rerank-GSM8k](https://huggingface.co/lqtrung1998/Codellama-7b-hf-SFT-Rerank-GSM8k)
* [Codellama-7b-hf-ReFT-GSM8k](https://huggingface.co/lqtrung1998/Codellama-7b-hf-ReFT-GSM8k)
* [Codellama-7b-hf-ReFT-Rerank-GSM8k](https://huggingface.co/lqtrung1998/Codellama-7b-hf-ReFT-Rerank-GSM8k)
* [galactica-6.7b-SFT-warmup-GSM8k](https://huggingface.co/lqtrung1998/galactica-6.7b-SFT-warmup-GSM8k)
* [galactica-6.7b-SFT-GSM8k](https://huggingface.co/lqtrung1998/galactica-6.7b-SFT-GSM8k)
* [galactica-6.7b-SFT-Rerank-GSM8k](https://huggingface.co/lqtrung1998/galactica-6.7b-SFT-Rerank-GSM8k)
* [galactica-6.7b-ReFT-GSM8k](https://huggingface.co/lqtrung1998/galactica-6.7b-ReFT-GSM8k)
* [galactica-6.7b-ReFT-Rerank-GSM8k](https://huggingface.co/lqtrung1998/galactica-6.7b-ReFT-Rerank-GSM8k)

Note: Our models are tuned based on Codellama and Galactica, thus, licenses applicable to Codellama and Galactica, such as [Llama license](https://github.com/lqtrung1998/mwp_ReFT/blob/main/Llama_License.txt) and non-commercial CC BY-NC 4.0 license, also hold on these models

## Evaluation Results
See evaluations results of the models at table 4 of the research paper.

Updated results:
|                                                                    |  Top-1 | Voting@100 | Rerank@100 |
|--------------------------------------------------------------------|:------:|:----------:|:----------:|
| Codellama-7b-hf-SFT-warmup-GSM8k                                   |  63.00 |      -     |      -     |
| Codellama-7b-hf-SFT-GSM8k<br>(+Codellama-7b-hf-SFT-Rerank-GSM8k)   | 63.68  |    68.0    |    77.0    |
| Codellama-7b-hf-ReFT-GSM8k<br>(+Codellama-7b-hf-ReFT-Rerank-GSM8k) | 75.28  |    78.0    |    81.2    |
| galactica-6.7b-SFT-warmup-GSM8k                                    |  48.37 |      -     |      -     |
| galactica-6.7b-SFT-GSM8k<br>(+galactica-6.7b-SFT-Rerank-GSM8k)     | 58.83  |    62.9    |    73.4    |
| galactica-6.7b-ReFT-GSM8k<br>(+galactica-6.7b-ReFT-Rerank-GSM8k)   |  68.91 |    71.9    |    76.4    |


## License:
[Apache2.0 License](https://github.com/lqtrung1998/mwp_ReFT/blob/main/License.txt)

## Citation
Please cite the paper if you use our data, model or code.
```
@inproceedings{luong2024reft,
      title={ReFT: Reasoning with Reinforced Fine-Tuning}, 
      author={Luong, Trung Quoc and Zhang, Xinbo and Jie, Zhanming and Sun, Peng and Jin, Xiaoran and Li, Hang},
      year={2024},
      booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
      url={https://arxiv.org/abs/2404.03592}
}
```
