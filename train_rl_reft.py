# Copyright 2023 Bytedance Ltd.
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import pad_across_processes, broadcast
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from datetime import timedelta
from functools import partial
import json
import os
import random
from src.python_engine import run_python_code, process_code
from src.utils import set_seed, floatify, compute_ETA, discount_cumsum, do_gather, allgather, allgather_masked_whiten
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from trl import AutoModelForCausalLMWithValueHead
from trl.core import masked_mean, masked_var, masked_whiten, logprobs_from_logits
import numpy as np
import wandb
import shutil
from prettytable import PrettyTable
tqdm = partial(tqdm, ncols=0, leave=False)

TIMEOUT = 10
instruction=None
cot_trigger=None
answer_trigger=None
def setup_cot(src_name):
    assert src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric']
    global instruction
    global cot_trigger
    global answer_trigger
    # Complete output is in this form: f'{instruction}{question.strip()}{cot_trigger}{answer_cot.strip()}'
    instruction = 'Question:\n'
    cot_trigger = '\nAnswer reasoning:\n'
    answer_trigger = '\nTherefore, the answer is: '
    return 

post_process_final_answer_fn_mapper = {
    'gsm8k': lambda x: float(x.replace(',','').strip()),
    'svamp': lambda x: float(x.replace(',','').strip()),
    'mathqa': lambda x: x.lower().replace('"','').replace("'",'').strip(),
    'mathqa-numeric': lambda x: float(x),
}
### the answer_cot is a list of answer_cot
post_process_answer_cot_fn_mapper = {
    ('python', 'gsm8k'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'svamp'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'mathqa'): lambda answer_cot: [str(res).lower().replace('"','').replace("'",'').strip() for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'mathqa-numeric'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('nl', 'gsm8k'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    ('nl', 'svamp'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    ('nl', 'mathqa'): lambda answer_cot: [res.split(answer_trigger)[-1].lower().replace('"','').replace("'",'').strip() for res in answer_cot],
    ('nl', 'mathqa-numeric'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
}
compare_answer_fn_mapper = {
    'gsm8k': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    'svamp': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    'mathqa': lambda extracted_ans, target_answer: extracted_ans == target_answer,
    'mathqa-numeric': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
}

def prepare_deepspeed_ref_model(model):
    # Adopted from: https://github.com/huggingface/trl/blob/02f5c1d8cee73045c837d01d7f1577a57779b035/trl/trainer/ppo_trainer.py#L1399
    import deepspeed

    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model

def prepare_datasets_and_data_loaders(args, tokenizer):
    with accelerator.main_process_first():
        # make raw dataset
        raw_dataset = DatasetDict({
            'train': Dataset.from_list(json.load(open(args['train_file'], 'r'))),
            'test': Dataset.from_list(json.load(open(args['test_file'], 'r'))),
        })
        accelerator.print('Raw data:', raw_dataset)

        # make cot related info
        src_name = raw_dataset['train']['item_id'][0].split('_')[0]  # e.g., gsm8k_0, gsm8k_1, gsm8k_2, ...
        setup_cot(src_name)
        accelerator.print('Using instruction:', instruction)
        accelerator.print('Using cot_trigger:', cot_trigger)
        accelerator.print('Using answer_trigger:', answer_trigger)
        def tokenize_fn(batch, args, tokenizer):
            assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
            new_batch = defaultdict(list)
            all_keys = list(batch.keys())
            for item_values in zip(*(batch[k] for k in all_keys)):
                item = {k: item_values[i] for i, k in enumerate(all_keys)}
                item_id, question, answer_value, answer_cot = \
                        item['item_id'], \
                        item['question'], \
                        item['answer_value'], \
                        item.get('answer_cot', None), \
                
                question = question.strip()
                if answer_value is not None:
                    answer_value = answer_value.strip()

                if answer_cot:
                    answer_cot = answer_cot.strip()
                    if args['engine'] == 'nl':
                        answer_cot += f'{answer_trigger}{answer_value}'

                input = f'{instruction}{question}{cot_trigger}'
                output = f'{answer_cot}'
                prefix_text = f'{instruction}{question}{cot_trigger}'

                # Modify for particular datasets and engine
                if src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric'] and args['engine'] == 'python':
                    prefix_text += f'def solution():\n    """{question}"""\n'

                input_encode = tokenizer(input, add_special_tokens=False)
                output_encode = tokenizer(output, add_special_tokens=False)
                prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

                input_ids = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
                labels = [-100] * len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]
                attention_mask = [1] * len(input_ids)
                prefix = prefix_encode['input_ids']
                prefix_attention_mask = prefix_encode['attention_mask']

                # Truncation
                input_ids = input_ids[:args['max_input_length']]
                labels = labels[:args['max_input_length']]
                attention_mask = attention_mask[:args['max_input_length']]
                prefix = prefix[:args['max_input_length']]
                prefix_attention_mask = prefix_attention_mask[:args['max_input_length']]

                ##
                new_batch['input_ids'].append(input_ids)
                new_batch['labels'].append(labels)
                new_batch['attention_mask'].append(attention_mask)
                new_batch['prefix'].append(prefix)
                new_batch['prefix_attention_mask'].append(prefix_attention_mask)
                ##
                new_batch['item_id'].append(item_id)
                new_batch['question'].append(question)
                new_batch['prefix_text'].append(prefix_text)
                new_batch['answer_cot'].append(answer_cot)
                new_batch['answer_value'].append(answer_value)

            return new_batch

        tokenized_dataset = DatasetDict({
            mode: dataset.map(
                tokenize_fn, fn_kwargs={'args': args, 'tokenizer': tokenizer}, batched=True,
                remove_columns=dataset.column_names,
                num_proc=None, load_from_cache_file=True, keep_in_memory=False,
            ) for mode, dataset in raw_dataset.items()})
        accelerator.print('Processed data:', tokenized_dataset)

        if accelerator.is_main_process and args['wandb_log']:
            wandb.config.update({
                "src_name": src_name,
                "instruction": instruction,
                "cot_trigger": cot_trigger,
                "answer_trigger": answer_trigger,
                "raw_dataset": str(raw_dataset),
                "tokenized_dataset": str(tokenized_dataset),
                # "train_input_ids_max_length": max(tokenized_dataset['train']['input_ids_max_length']),
                # "test_input_ids_max_length": max(tokenized_dataset['test']['input_ids_max_length']),
            })

    def collate_fn(batch, args, tokenizer):
        max_input_length = max([len(item['input_ids']) for item in batch])
        max_target_length = max([len(item['labels']) for item in batch])
        max_prefix_length = max([len(item['prefix']) for item in batch])

        input_ids, input_ids_left_padded = [], []
        attention_mask, attention_mask_left_padded = [], []
        labels, labels_left_padded = [], []
        prefix, prefix_left_padded = [], []
        prefix_attention_mask, prefix_attention_mask_left_padded = [], []

        for item in batch:
            # input_ids.append(item['input_ids'] + [tokenizer.pad_token_id] * (max_input_length - len(item['input_ids'])))
            # attention_mask.append(item['attention_mask'] + [0] * (max_input_length - len(item['attention_mask'])))
            # labels.append(item['labels'] + [-100] * (max_target_length - len(item['labels'])))

            labels_left_padded.append([-100] * (max_target_length - len(item['labels'])) + item['labels'])
            prefix_left_padded.append([tokenizer.pad_token_id] * (max_prefix_length - len(item['prefix'])) + item['prefix'])
            prefix_attention_mask_left_padded.append(
                [0] * (max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask'])

        ppo_forward_kwargs = {
            'query': [item['prefix_text'] for item in batch],
            'query_tensors': torch.LongTensor(prefix_left_padded),
            'query_tensors_attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
            # 'answer_values': torch.FloatTensor([float(item['answer_value'].replace(',', '')) for item in batch]),
            'answer_values': [item['answer_value'].replace(',', '') for item in batch],
            'item_ids': torch.LongTensor([int(item['item_id'].split('_')[1]) for item in batch]),
            # 'answer_cot': [item['answer_cot'] for item in batch],
            # 'sft_model_input_ids': torch.LongTensor(input_ids),
            # 'sft_model_attention_mask': torch.BoolTensor(attention_mask),
            # 'sft_model_labels': torch.LongTensor(labels),
        }
        generate_prefix_kwargs = {
            'input_ids': torch.LongTensor(prefix_left_padded),
            'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
            'labels': torch.LongTensor(labels_left_padded)
        }

        return {
            'ppo_forward_kwargs': ppo_forward_kwargs,
            'generate_prefix_kwargs': generate_prefix_kwargs,
        }

    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=args['batch_size'],
                                  num_workers=args['num_workers'], pin_memory=True,
                                  collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))

    test_dataloader = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=args['eval_batch_size'],
                                 num_workers=args['num_workers'], pin_memory=True,
                                 collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))

    return (tokenized_dataset['train'], train_dataloader), (tokenized_dataset['test'], test_dataloader)

def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
    os.makedirs(save_path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    tokenizer.save_pretrained(save_path)
    if accelerator.is_main_process and most_recent_ckpts_paths is not None:
        most_recent_ckpts_paths.append(save_path)
        if args['keep_num_ckpt'] is not None and len(most_recent_ckpts_paths) > args['keep_num_ckpt']:
            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
            # os.remove(ckpt_to_be_removed)
            shutil.rmtree(ckpt_to_be_removed)

def rollout(args, model, ref_model, tokenizer, query_tensors, query_tensors_attention_mask, answer_values, src_name):
    model.eval()
    with torch.no_grad():
        gen_output = accelerator.unwrap_model(model).generate(
            input_ids=query_tensors,
            attention_mask=query_tensors_attention_mask,
            top_k=0.0, top_p=1.0,
            do_sample=True,
            # output_scores=True,
            # return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            # bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args['max_gen_length'],
        )
        # completed_tensors, logits_per_steps = gen_output[0], gen_output[1]
        completed_tensors = gen_output
        completed_tensors = pad_across_processes(completed_tensors, dim=1, pad_index=tokenizer.pad_token_id, pad_first=False)

    # Evaluate score
    completed_texts = tokenizer.batch_decode(completed_tensors.cpu().numpy().tolist(), skip_special_tokens=True)
    programs = [text.strip().split(cot_trigger)[-1].strip() for text in completed_texts]
    execute_fn = post_process_answer_cot_fn_mapper[(args['engine'], src_name)]
    correctness = []
    for i, extracted_ans in enumerate(execute_fn(programs)):
        target_value = post_process_final_answer_fn_mapper[src_name](answer_values[i])
        if extracted_ans is not None:
            if args['engine'] == 'game24' or args['engine'] == 'calcn':
                is_correct = extracted_ans
            else:
                if compare_answer_fn_mapper[src_name](extracted_ans, target_value):
                    is_correct = 1
                else:
                    is_correct = 0.1
                    # for mathqa, even though it can executed, if the results is not within a,b,c,d,xxx, still zero reward
                    # because we want to give some reward for the prediction that able to select one of the answer
                    # for example, the executed answer is "{}" in mathqa.
                    # THIS PART IS TO BE DECIDED.
                    # if src_name == 'mathqa' and not (len(extracted_ans) == 1 and extracted_ans.isalpha()):
                    #     is_correct = 0
        else:
            is_correct = 0
        correctness.append(is_correct)

    model_input_ids = completed_tensors
    model_attention_mask = (completed_tensors != tokenizer.pad_token_id)
    with torch.no_grad():
        # Get old logprob and val
        lm_logits, _dummy2, val = model(input_ids=model_input_ids, attention_mask=model_attention_mask)
        old_logprob = logprobs_from_logits(lm_logits[:, :-1, :], labels=model_input_ids[:, 1:])  # (bs, seqlen-1)

        # Get the ref model logprob
        ref_logprob = None
        if ref_model is not None:
            ref_lm_logits, _dummy2, _dummy3 = ref_model(input_ids=model_input_ids, attention_mask=model_attention_mask)
            ref_logprob = logprobs_from_logits(ref_lm_logits[:, :-1, :], labels=model_input_ids[:, 1:])  # (bs, seqlen-1)

    # Masking the last prompt token up untils the token before eos_token_id
    prompt_len = query_tensors.size(1)
    mask = torch.zeros_like(model_input_ids, dtype=torch.bool)  # (bs, seqlen)
    mask[:, query_tensors.size(1) - 1: -1] = 1
    score_rew = np.zeros(mask.shape)  # (bs, seqlen)
    score_rew[:, -2] = np.array(correctness)
    nonzero = (model_input_ids == tokenizer.eos_token_id).nonzero()
    for (bidx, tidx) in nonzero:
        mask[bidx][tidx:] = 0
        score_rew[bidx][tidx:] = 0
        score_rew[bidx][tidx - 1] = correctness[bidx]

    # Make the kl reward and the full reward
    kl_rew = None
    rew = score_rew
    if ref_logprob is not None:
        kl = old_logprob - ref_logprob  # (bs, seqlen-1)
        kl = (kl.float() * mask[:, :-1]).cpu().numpy()
        kl_rew = np.zeros(mask.shape)  # (bs, seqlen)
        kl_rew[:, :-1] = -kl # NOTE the minus sign
 
        kl_coef = args["kl_coef"]
        rew = score_rew + kl_coef * kl_rew

    # Process val ret adv logprob
    val = (val.float() * mask).cpu().numpy()
    gamma = args["gamma"]
    lam = args["lam"]
    # ret = np.zeros_like(rew)
    adv = np.zeros_like(rew)
    for i in range(len(rew)):
        cur_rew, cur_val = rew[i], val[i]
        cur_delta = -cur_val[:-1] + cur_rew[:-1] + gamma * cur_val[1:]
        cur_adv = discount_cumsum(cur_delta, discount=gamma * lam)
        cur_adv[:prompt_len - 1] = 0
        adv[i][:-1] = cur_adv

    # lambda_return = GAE + values
    ret = adv + val  # (bs, seqlen)

    rew = torch.tensor(rew, device=mask.device, dtype=old_logprob.dtype) * mask
    score_rew = torch.tensor(score_rew, device=mask.device, dtype=old_logprob.dtype) * mask
    if kl_rew is not None:
        kl_rew = torch.tensor(kl_rew, device=mask.device, dtype=old_logprob.dtype) * mask
    ret = torch.tensor(ret, device=mask.device, dtype=old_logprob.dtype) * mask
    val = torch.tensor(val, device=mask.device, dtype=old_logprob.dtype) * mask
    adv = torch.tensor(adv, device=mask.device, dtype=old_logprob.dtype) * mask
    old_logprob = old_logprob * mask[:, :-1]

    ## Debug
    # accelerator.print("padded_prompt_len:", prompt_len)
    # accelerator.print("model_input_ids:", tokenizer.batch_decode(model_input_ids[:1].cpu().numpy().tolist()))
    # accelerator.print("model_attention_mask:", model_attention_mask[:1].cpu().float().numpy().tolist())
    # accelerator.print("mask:", mask[:1].cpu().float().numpy().tolist())
    # accelerator.print("rew:", rew[:1].cpu().float().numpy().tolist())
    # accelerator.print("ret:", ret[:1].cpu().float().numpy().tolist())
    # accelerator.print("val:", val[:1].cpu().float().numpy().tolist())
    # accelerator.print("adv:", adv[:1].cpu().float().numpy().tolist())
    # accelerator.print("old_logprob:", old_logprob[:1].cpu().float().numpy().tolist())

    model.train()
    return model_input_ids, model_attention_mask, mask, rew, score_rew, kl_rew, ret, correctness, val, old_logprob, ref_logprob, adv

def train_one_epoch(args, model, ref_model, train_dataset, train_dataloader, optimizer, scheduler, tokenizer,
                    global_step, global_iter_num, test_dataset, test_dataloader,
                    prefix, epoch, best_eval_log_dict, summary_log_dict, most_recent_ckpts_paths):
    model_dir = args['model_dir']
    clip_grad_norm = args.get('clip_grad_norm', None)
    vf_coef = args['vf_coef']
    evaluating_step_freq = args.get('evaluating_step_freq', None)
    logging_step_freq = args.get('logging_step_freq', None)
    saving_step_freq = args.get('saving_step_freq', None)
    model.train()
    epoch_result_dict = defaultdict(list)
    with tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not accelerator.is_main_process, desc='Train Loop') as t:
        for idx, batch in t:
            result_dict = defaultdict(list)
            # Do rollout first
            model.eval()
            model_input_ids, model_attention_mask, mask, rew, score_rew, kl_rew, ret, correctness, val, old_logprob, ref_logprob, adv = rollout(
                args, model, ref_model, tokenizer,
                query_tensors=batch['ppo_forward_kwargs']['query_tensors'],
                query_tensors_attention_mask=batch['ppo_forward_kwargs']['query_tensors_attention_mask'],
                answer_values=batch['ppo_forward_kwargs']['answer_values'],
                src_name=train_dataset[0]['item_id'].split('_')[0],
            )
            model.train()
            # preprocess
            raw_adv = adv
            if args['adv_whitening'] == 'global':
                adv = allgather_masked_whiten(adv, mask) # (mini_bs, seqlen)
            elif args['adv_whitening'] == 'local':
                adv = masked_whiten(adv, mask)

            batch_size_per_gpu = len(batch['ppo_forward_kwargs']['query'])
            mini_batch_size_per_gpu = args["mini_batch_size"]
            ppo_epochs = args["ppo_epochs"]
            train_stats = {}
            for _ in range(ppo_epochs):
                perms = torch.randperm(batch_size_per_gpu)
                for mini_idx in range(0, len(perms), mini_batch_size_per_gpu):
                    b_inds = perms[mini_idx: mini_idx + mini_batch_size_per_gpu]
                    # Subset to batch
                    cur_val = val[b_inds].contiguous()  # mini_bs x seqlen
                    cur_old_logprob = old_logprob[b_inds].contiguous()  # mini_bs x seqlen
                    cur_mask = mask[b_inds].contiguous()  # mini_bs x seqlen
                    cur_rew = rew[b_inds].contiguous()  # mini_bs x seqlen
                    cur_score_rew = score_rew[b_inds].contiguous() # mini_bs x seqlen
                    cur_kl_rew = None if kl_rew is None else kl_rew[b_inds].contiguous()  # mini_bs x seqlen
                    cur_ret = ret[b_inds].contiguous()  # mini_bs x seqlen
                    cur_adv = adv[b_inds].contiguous()  # mini_bs x seqlen
                    cur_raw_adv = raw_adv[b_inds].contiguous()  # mini_bs x seqlen
                    cur_model_input_ids = model_input_ids[b_inds].contiguous()  # mini_bs x seqlen
                    cur_model_attention_mask = model_attention_mask[b_inds].contiguous()  # mini_bs x seqlen
                    
                    resp_len_per_sample = torch.clamp(torch.sum(cur_mask, dim=1), min=1.0)  # (mini_bs,)
                    cur_query_mask = torch.logical_xor(cur_mask, cur_model_attention_mask)  # (mini_bs, seqlen)
                    query_len_per_sample = torch.clamp(torch.sum(cur_query_mask, dim=1), min=1.0)  # (mini_bs,)

                    # Preprocess advantage and get metrics  
                    cur_mask = cur_mask.type(cur_adv.dtype).contiguous()
                    mean_adv, var_adv = masked_mean(cur_adv, cur_mask), masked_var(cur_adv, cur_mask)

                    # Forward current model
                    model.eval()
                    lm_logits, _, vpreds = model(input_ids=cur_model_input_ids, attention_mask=cur_model_attention_mask)
                    logprob = logprobs_from_logits(lm_logits[:, :-1, :], cur_model_input_ids[:, 1:])  # (mini_bs, seqlen-1)

                    # Compute losses
                    loss = 0

                    # policy gradient loss
                    ratio = torch.exp(logprob - cur_old_logprob)
                    pg_losses = -cur_adv[:, :-1] * ratio
                    pg_losses2 = -cur_adv[:, :-1] * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                    pg_loss = ((torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum(dim=-1) / resp_len_per_sample).mean()
                    # pg_loss = (torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum() / cur_mask[:, :-1].sum()
                    # pg_loss = (-logprob * cur_ret[:,:-1]).sum() / cur_mask[:, :-1].sum()

                    # value loss
                    vpredclipped = torch.max(torch.min(vpreds, cur_val + 0.2), cur_val - 0.2)
                    vf_losses1 = (vpreds - cur_ret) ** 2
                    vf_losses2 = (vpredclipped - cur_ret) ** 2
                    vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum(dim=-1) / resp_len_per_sample).mean()
                    # vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum() / cur_mask.sum())

                    # total loss
                    loss += pg_loss + vf_coef * vf_loss

                    # model_output = model(input_ids=model_input_ids, attention_mask=model_attention_mask)
                    # logits = model_output[0]
                    # logprob_dist = torch.nn.functional.log_softmax(logits,dim=-1)
                    # logprob = torch.gather(logprob_dist, 2, model_input_ids[:, 1:].unsqueeze(2)).squeeze(-1)
                    # loss_pg = (-logprob * ret[:,:-1]).sum() / torch.maximum(torch.sum(mask[:,:-1]), torch.tensor(1.0))
                    # loss += loss_pg

                    # sft_model_input_ids = batch['ppo_forward_kwargs']['sft_model_input_ids']
                    # sft_model_attention_mask = batch['ppo_forward_kwargs']['sft_model_attention_mask']
                    # sft_model_labels = batch['ppo_forward_kwargs']['sft_model_labels']
                    # loss_sft = model(input_ids=sft_model_input_ids, attention_mask=sft_model_attention_mask, labels=sft_model_labels)[0]
                    # loss += loss_sft

                    # token related metrics
                    mean_query_len = torch.mean(allgather(torch.mean(query_len_per_sample)))
                    std_query_len = torch.mean(allgather(torch.std(query_len_per_sample)))
                    mean_resp_len = torch.mean(allgather(torch.mean(resp_len_per_sample)))
                    std_resp_len = torch.mean(allgather(torch.std(resp_len_per_sample)))

                    # value related metrics
                    # vf_expl_var_num = torch.var(torch.masked_select(cur_ret - vpreds, cur_mask.bool())) 
                    # vf_expl_var_dem = torch.var(torch.masked_select(cur_ret, cur_mask.bool()))
                    vf_expl_var_num = masked_var(cur_ret - vpreds, cur_mask)
                    vf_expl_var_dem = masked_var(cur_ret, cur_mask)
                    vf_expl_var = 1.0 - vf_expl_var_num / (vf_expl_var_dem + 1e-8)
                    vf_expl_var = max(-1.0, vf_expl_var.item())  # the truncated value suffices
                    mean_vpred = masked_mean(vpreds, cur_mask)
                    mean_return = masked_mean(cur_ret, cur_mask)
                    mean_reward = masked_mean(cur_rew, cur_mask)
                    mean_score_reward = masked_mean(cur_score_rew, cur_mask)
                    mean_kl_reward = 0.0 if cur_kl_rew is None else masked_mean(cur_kl_rew, cur_mask)
                    mean_kcxkl_reward = args["kl_coef"] * mean_kl_reward

                    # policy related metrics
                    mean_ratio = masked_mean(ratio, cur_mask[:, :-1])
                    #mean_adv = masked_mean(cur_adv[:, :-1], cur_mask[:, :-1])
                    mean_logprob = masked_mean(logprob, cur_mask[:, :-1])
                    # sequence-level kl
                    mean_seq_kl = -1.0
                    if cur_kl_rew is not None:
                        cur_kl = -cur_kl_rew
                        seq_kl = torch.sum(cur_kl * cur_mask, dim=1)  # (mini_bs,)
                        mean_seq_kl = torch.mean(seq_kl)

                    # Update
                    epoch_result_dict['loss'].append(loss.item())

                    # accelerator.backward(loss)
                    # accelerator.deepspeed_engine_wrapped.backward(loss)
                    # runs backpropagation and handles mixed precision
                    if accelerator.distributed_type == "DEEPSPEED":
                        accelerator.deepspeed_engine_wrapped.engine.backward(loss)
                        total_grad_norm = 0.0
                        for n, p in model.named_parameters():
                            cur_grad = deepspeed.utils.safe_get_full_grad(p).view(-1)
                            cur_grad_norm_sqrt = torch.norm(cur_grad, 2)
                            if cur_grad_norm_sqrt < 1e-8:
                                accelerator.print(f'{n} grad_norm_sqrt: {cur_grad_norm_sqrt}')
                            total_grad_norm += cur_grad_norm_sqrt ** 2
                        total_grad_norm = total_grad_norm ** 0.5
                        # Deepspeed's `engine.step` performs the following operations:
                        # - gradient accumulation check
                        # - gradient clipping
                        # - optimizer step
                        # - zero grad
                        # - checking overflow
                        # - lr_scheduler step (only if engine.lr_scheduler is not None)
                        accelerator.deepspeed_engine_wrapped.engine.step()
                    else:
                        accelerator.backward(loss)
                        #accelerator.backward(loss)
                        total_grad_norm = -1.0
                        if clip_grad_norm is not None:
                            total_grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                    # Update running stats
                    n_correct, total = do_gather([sum(correctness), len(correctness)])
                    train_stats["acc"] = n_correct / total
                    train_stats["ncor"] = n_correct
                    train_stats["total"] = total
                    train_stats['pg_loss'] = pg_loss.item()
                    train_stats['vf_loss'] = vf_loss.item()
                    train_stats['vf_expl_var'] = vf_expl_var

                    for k, v in train_stats.items():
                        result_dict[k].append(v)

                    total_param_norm = 0.0
                    if accelerator.distributed_type == "DEEPSPEED":
                        for n, p in model.named_parameters():
                            cur_param = deepspeed.utils.safe_get_full_fp32_param(p).view(-1)
                            total_param_norm += torch.norm(cur_param, 2) ** 2
                        total_param_norm = total_param_norm ** 0.5
                    else:
                        total_param_norm = torch.norm(
                            torch.cat([p.view(-1) for p in model.parameters()]),
                            p=2  # L2 norm
                        )
                    # logging
                    if accelerator.is_main_process and args['wandb_log']:
                        wandb.log({
                            "nn/total_grad_norm": total_grad_norm,
                            "nn/total_param_norm": total_param_norm,
                            "nn/lr": scheduler.get_last_lr()[0],
                        }, step=global_iter_num)
                        wandb.log({
                            "acc/acc": train_stats["acc"],
                            "acc/ncor": train_stats["ncor"],
                            "acc/total": train_stats["total"],
                        }, step=global_iter_num)
                        wandb.log({
                            "loss/loss:": loss,
                            "loss/pg_loss": pg_loss,
                            "loss/vf_loss": vf_loss,
                        }, step=global_iter_num)
                        wandb.log({
                            "tokens/mean_query_len": mean_query_len,
                            "tokens/std_query_len": std_query_len,
                            "tokens/mean_resp_len": mean_resp_len,
                            "tokens/std_resp_len": std_resp_len,
                        }, step=global_iter_num)
                        wandb.log({
                            "policy/mean_ratio": mean_ratio,
                            "policy/mean_adv": mean_adv,
                            "policy/var_adv": var_adv,
                            "policy/mean_logprob": mean_logprob,
                            "policy/mean_seq_kl": mean_seq_kl,
                        }, step=global_iter_num)
                        wandb.log({
                            "value/vf_expl_var": vf_expl_var,
                            "value/mean_vpred": mean_vpred,
                            "value/mean_return": mean_return,
                            "value/mean_reward": mean_reward,
                            "value/mean_score_reward": mean_score_reward,
                            "value/mean_kl_reward": mean_kl_reward,
                            "value/mean_kcxkl_reward": mean_kcxkl_reward,
                        }, step=global_iter_num)
                    # Update iter num
                    # torch.distributed.barrier()
                    global_iter_num += 1

            scheduler.step()
            global_step += 1
            # accelerator.empty_cache()
            # Step update metric
            epoch_result_dict['loss'].append(loss.item())
            for k, v in train_stats.items():
                epoch_result_dict[k].append(v)

            # Step evaluating
            eval_log_dict = {}
            is_best = False
            if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
                evaluate_result_dict = {f'Eval.Gen.{k}': v for k, v in
                                        evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer).items()}
                eval_log_dict.update(evaluate_result_dict)
                if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', -1):
                    is_best = True
                    best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
                    if 'Eval.Gen.value_accuracy' not in summary_log_dict:
                        summary_log_dict['Eval.Gen.value_accuracy'] = []
                    summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])

            # Step logging
            train_log_dict = {}
            if logging_step_freq is not None and global_step % logging_step_freq == 0:
                train_log_dict = {f'T.{k}': sum(v) / len(v) if isinstance(v, list) else v for k, v in epoch_result_dict.items()}

            if eval_log_dict or train_log_dict:
                log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
                if accelerator.is_main_process and args['wandb_log']:
                    wandb.log(log_dict, step=global_step)
                    log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'], **log_dict}
                log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k,v in log_dict.items()}
                accelerator.print(f"{prefix}[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}")

            # Step saving
            if saving_step_freq is not None and global_step % saving_step_freq == 0:
                if is_best:
                    save_path = os.path.join(model_dir, f'best')
                    do_checkpoint(args, model, tokenizer, save_path)
                if args['keep_num_ckpt'] > 0:
                    save_path = os.path.join(model_dir, f'global_step_{str(global_step)}')
                    do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)

            # Keep only max_record items
            for k, v in epoch_result_dict.items():
                if len(v) > 1:
                    epoch_result_dict[k] = v[-1:]

    # Metric summary:
    epoch_result_dict = {k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in epoch_result_dict.items()}
    return epoch_result_dict, global_step, global_iter_num

def evaluate_generation(args, model, dataset, dataloader, tokenizer):
    model.eval()
    predictions = []
    targets = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process,
                           desc='Evaluation Gen Loop'):
        output_ = accelerator.unwrap_model(model).generate(
            **batch['generate_prefix_kwargs'],
            max_length=args['max_gen_length'],
            output_scores=True,
            return_dict_in_generate=True,
            num_beams=1,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_.sequences
        generated_ids = pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)

        labels = batch['generate_prefix_kwargs']['labels']
        labels = pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        labels[labels == -100] = tokenizer.pad_token_id

        generated_ids, labels = accelerator.gather(generated_ids), accelerator.gather(labels)

        preds = [tokenizer.decode(g.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in
                 generated_ids]
        predictions.extend(preds)
        target = [tokenizer.decode(t.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for t in
                  labels]
        targets.extend(target)

    predictions = predictions[:len(dataset)]
    targets = targets[:len(dataset)]

    if accelerator.is_main_process and accelerator.is_local_main_process:
        results = []
        src_name = dataset[0]['item_id'].split('_')[0]
        for pred, tar, item in zip(predictions, targets, dataset):
            cur_res = {
                'item_id': item['item_id'],
                'answer_value': item['answer_value'],
            }
            ## Processing target
            target_cot = tar.strip().split(cot_trigger)[-1].strip()
            target_value = post_process_final_answer_fn_mapper[src_name](cur_res['answer_value'])
            cur_res['target'] = target
            cur_res['target_cot'] = target_cot
            cur_res['target_value'] = target_value
            ## Processing prediction
            prediction_cot = pred.strip().split(cot_trigger)[-1].strip()
            cur_res['prediction'] = pred
            cur_res['prediction_cot'] = prediction_cot
            cur_res['prediction_value'] = None # Tobe filled
            results.append(cur_res)

        execute_fn = post_process_answer_cot_fn_mapper[(args['engine'], src_name)]
        corr_value = 0
        for i, prediction_value in enumerate(execute_fn([item['prediction_cot'] for item in results])):
            target_value = results[i]['target_value']
            is_correct = compare_answer_fn_mapper[src_name](prediction_value, target_value) if prediction_value is not None else False
            results[i]['prediction_value'] = prediction_value
            results[i]['is_correct'] = is_correct
            corr_value += is_correct

        res_path = args['model_dir'].rstrip('/')+ '/' + '_res.json'
        with open(res_path, 'w') as f:
            json.dump(results, f, indent=2)

        # if args['wandb_log']:
        #     table = wandb.Table(dataframe=pd.DataFrame(results))
        #     wandb.log({"predictions": table})

        value_accuracy = corr_value / len(results) * 100
        accelerator.print(f"[Eval Info] value_accuracy: {value_accuracy:.5g}%")
        value_accuracy = torch.FloatTensor([value_accuracy]).to(accelerator.device)
    else:
        value_accuracy = torch.FloatTensor([-1.0]).to(accelerator.device)
    value_accuracy = broadcast(value_accuracy).cpu().numpy().tolist()[0]

    # Metric summary:
    model.train()
    return {'value_accuracy': value_accuracy}

def main(args):
    set_seed(args['seed'] + accelerator.process_index)

    if accelerator.is_main_process and args['wandb_log']:
        wandb.init(project=args['wandb_project'], name=args['wandb_run_name'])
        wandb.config.update(args)
        
    tokenizer = AutoTokenizer.from_pretrained(args['tokenizer_name_or_path'], use_fast=True)
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 2

    (train_dataset, train_dataloader), (test_dataset, test_dataloader) = prepare_datasets_and_data_loaders(args, tokenizer)

    MODEL_CLASS = AutoModelForCausalLMWithValueHead
    model = MODEL_CLASS.from_pretrained(args['model_name_or_path'])
    # accelerator.print(f'[Vocab size]: {len(tokenizer)}')
    # model.resize_token_embeddings(len(tokenizer))

    # initialize ref model (if any)
    ref_model = None
    if args['ref_model_name_or_path']:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args['ref_model_name_or_path'])
        # from copy import deepcopy
        # ref_model = deepcopy(model)

    # optimizer
    n_epochs = args['n_epochs']
    num_training_steps = (len(train_dataloader) // accelerator.num_processes * n_epochs)
    warmup_step = args['warmup_step'] if args['warmup_step'] is not None and args['warmup_step'] >= 0 else int(0.1 * num_training_steps)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": args['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=1e-8)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                              test_dataloader)
    if ref_model is not None:
        if accelerator.distributed_type == "DEEPSPEED":
            ref_model = prepare_deepspeed_ref_model(ref_model)
        else:
            ref_model = accelerator.prepare(ref_model)

    global_step = 0
    global_iter_num = 0
    evaluating_epoch_freq = args['evaluating_epoch_freq']
    logging_epoch_freq = args['logging_epoch_freq']
    saving_epoch_freq = args['saving_epoch_freq']
    model_dir = args['model_dir']
    best_eval_log_dict = {}
    summary_log_dict = {}
    os.makedirs(model_dir, exist_ok=True)
    most_recent_ckpts_paths = []
    with tqdm(range(1, n_epochs+1), total=n_epochs, disable=False) as t:
        for epoch in t:
            kwargs = {
                'args': args,
                'model': model,
                'ref_model': ref_model,
                'train_dataset': train_dataset,
                'train_dataloader': train_dataloader,
                'test_dataset': test_dataset,
                'test_dataloader': test_dataloader,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'global_step': global_step,
                'global_iter_num': global_iter_num,
                'tokenizer': tokenizer,
                'prefix': '',
                'epoch': epoch,
                'best_eval_log_dict': best_eval_log_dict,
                'summary_log_dict': summary_log_dict,
                'most_recent_ckpts_paths': most_recent_ckpts_paths,
            }
            train_epoch_result_dict, global_step, global_iter_num = train_one_epoch(**kwargs)

            eval_log_dict = {}
            is_best = False
            if evaluating_epoch_freq is not None and epoch % evaluating_epoch_freq == 0:
                evaluate_result_dict = {f'Eval.Gen.{k}': v for k, v in
                                        evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer).items()}
                eval_log_dict.update(evaluate_result_dict)
                if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', -1):
                    is_best = True
                    best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
                    if 'Eval.Gen.value_accuracy' not in summary_log_dict:
                        summary_log_dict['Eval.Gen.value_accuracy'] = []
                    summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])

            train_log_dict = {}
            if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
                train_log_dict = {f'T.{k}': sum(v) / len(v) if isinstance(v, list) else v for k, v in
                                train_epoch_result_dict.items()}

            if eval_log_dict or train_log_dict:
                log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
                if accelerator.is_main_process and args['wandb_log']:
                    wandb.log(log_dict, step=global_iter_num)
                    log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'] + '|' + wandb.run.id, **log_dict}

                log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k, v in log_dict.items()}
                accelerator.print(
                    f"[Epoch={epoch}/{args['n_epochs']}, Step={global_step}] {log_dict}")

            if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
                if is_best:
                    save_path = os.path.join(model_dir, f'best')
                    do_checkpoint(args, model, tokenizer, save_path)
                #
                if args['keep_num_ckpt'] > 0:
                    # save the checkpoint only if keep num ckpt > 0
                    save_path = os.path.join(args['model_dir'], f'global_step_{str(global_step)}_epoch_{epoch}')
                    do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)

    return 

if __name__ == '__main__':
    from transformers import HfArgumentParser

    NONE_INT = -100
    NONE_STR = 'None'


    @dataclass
    class Arguments:
        model_name_or_path: str
        tokenizer_name_or_path: str
        model_dir: str
        train_file: str
        test_file: str
        batch_size: int = field(default=8)
        mini_batch_size: int = field(default=8)
        eval_batch_size: int = field(default=8)
        ppo_epochs: int = field(default=1)
        n_epochs: int = field(default=40)
        num_workers: int = field(default=8)
        learning_rate: float = field(default=2e-5)
        weight_decay: float = field(default=1e-6)
        warmup_step: int = field(default=0)
        clip_grad_norm: float = field(default=1)
        vf_coef: float = field(default=1.0)
        kl_coef: float = field(default=0.1)
        gamma: float = field(default=0.98)
        lam: float = field(default=0.95)
        ref_model_name_or_path: str = field(default="")
        evaluating_epoch_freq: int = field(default=1)
        logging_epoch_freq: int = field(default=1)
        saving_epoch_freq: int = field(default=1000)
        evaluating_step_freq: int = field(default=NONE_INT)
        logging_step_freq: int = field(default=NONE_INT)
        logging_seq_str_step_freq: int = field(default=NONE_INT)
        logging_values_step_freq: int = field(default=NONE_INT)
        saving_step_freq: int = field(default=NONE_INT)
        seed: int = field(default=42)
        max_input_length: int = field(default=700)
        max_gen_length: int = field(default=700)
        keep_num_ckpt: int = field(default=5)
        # wandb stuff
        wandb_log: bool = field(default=False)
        wandb_project: str = field(default='tmp_anvfupsadfn')
        wandb_run_name: str = field(default='default_run_name')
        ###
        engine: str = field(default='python')
        adv_whitening: str = field(default='global')

    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)
    for k, v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))])
    accelerator.print(args)
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    main(args)
