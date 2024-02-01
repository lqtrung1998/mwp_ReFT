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
from collections import Counter
from dataclasses import dataclass, field, asdict
from datasets import Dataset
from datetime import timedelta
import deepspeed
from functools import partial
import json
import os
from src.python_engine import run_python_code
from src.utils import set_seed, write_data, floatify
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Dict
import wandb
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

def tokenize_fn(examples: Dict, tokenizer, max_length, src_name, engine):
    features = {"input_ids": [], "attention_mask": [], "answer_value": [], "answer_cot": [], "question": [], 'item_id': []}
    for idx, question in enumerate(examples["question"]):
        text = f"{instruction}{question}{cot_trigger}"
        if src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric'] and engine == 'python':
            text += f'def solution():\n    """{question}"""\n'
        source_text_res = tokenizer.encode_plus(text, max_length=max_length, truncation=True, add_special_tokens=False)
        features["input_ids"].append(source_text_res["input_ids"])
        features["attention_mask"].append(source_text_res["attention_mask"])
        features["question"].append(question)
        features["answer_value"].append(examples["answer_value"][idx])
        features["answer_cot"].append(None if "answer_cot" not in examples else examples["answer_cot"][idx])
        features['item_id'].append(examples['item_id'][idx])
    return features

def collate_fn(features, tokenizer):
    batch = {"input_ids": [], "attention_mask": []}
    max_input_length = max(len(x["input_ids"]) for x in features)
    for feature in features:
        input_ids = feature["input_ids"]
        attention_mask = feature["attention_mask"]
        input_ids = [tokenizer.pad_token_id] * (max_input_length - len(input_ids)) + input_ids
        attention_mask = [0] * (max_input_length - len(attention_mask)) + attention_mask
        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append(attention_mask)
    batch["input_ids"] = torch.tensor(batch["input_ids"])
    batch["attention_mask"] = torch.tensor(batch["attention_mask"])
    return batch

def main(args):
    set_seed(42)
    if torch.distributed.get_rank() == 0 and args['wandb_log']:
        wandb.init(project=args['wandb_project'], name=args['wandb_run_name'])
        wandb.config.update(args)
    # Init parameter
    model_name = args['model_name'] 
    input_path = args['input_path']
    save_dir = args['save_dir']
    engine = args['engine']
    batch_size = args['batch_size']
    max_length = args['max_length']
    num_return_sequences = args['num_return_sequences']
    temperature = args['temperature']
    do_sample = args['do_sample']

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 2

    # loading training data
    raw_dataset = Dataset.from_list(json.load(open(input_path,'r')))
    accelerator.print('Raw data:', raw_dataset)
    src_name = raw_dataset[0]['item_id'].split('_')[0]  # e.g., gsm8k_0, gsm8k_1, gsm8k_2, ...
    setup_cot(src_name)
    accelerator.print('Using instruction:', instruction)
    accelerator.print('Using cot_trigger:', cot_trigger)
    accelerator.print('Using answer_trigger:', answer_trigger)
    tokenized_dataset = raw_dataset.map(
        tokenize_fn, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'src_name': src_name, 'engine': engine},
        batched=True, remove_columns=raw_dataset.column_names, load_from_cache_file=False, num_proc=8
    )
    valid_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=partial(collate_fn, tokenizer=tokenizer))

    # loading model
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = model.eval()
    inf_config = {
        "replace_with_kernel_inject": False,
        "dtype": torch.bfloat16,
        "enable_cuda_graph": False,
        "tensor_parallel": {"tp_size": 8},
        'max_out_tokens': 1024,
        'min_out_tokens': 1,
    }
    model = deepspeed.init_inference(model=model, config=inf_config)
    all_results = []
    acc = 0
    for b_idx, batch in tqdm(enumerate(valid_dataloader), desc="Generating", total=len(valid_dataloader)):
        with torch.no_grad():
            # model.module.reset_cache()
            outputs = model.module.generate(batch["input_ids"].to(torch.cuda.current_device()),
                                    attention_mask=batch["attention_mask"].to(torch.cuda.current_device()),
                                    do_sample=do_sample, 
                                    max_length=max_length,
                                    num_return_sequences=num_return_sequences,
                                    use_cache=True,
                                    temperature=temperature,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id) ## batch_size x num_return_sequence, sequence length
            cur_batch_size = len(outputs)//num_return_sequences
            if accelerator.is_main_process:
                decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                cur_idx = len(all_results)
                for b_item_idx in range(cur_batch_size):
                    if cur_idx + b_item_idx >= len(tokenized_dataset):
                        break
                    obj = tokenized_dataset[cur_idx+b_item_idx]
                    new_obj = {
                        "item_id": obj["item_id"],
                        "question": obj["question"],
                        "answer_value": obj["answer_value"],
                        "answer_cot": obj["answer_cot"],
                        "predictions": [],
                    }
                    execute_fn = post_process_answer_cot_fn_mapper[(args['engine'], src_name)]
                    target_value = post_process_final_answer_fn_mapper[src_name](obj['answer_value'])
                    prediction_cots = [pred.strip().split(cot_trigger)[-1] for pred in decoded_output[b_item_idx*num_return_sequences: (b_item_idx+1)*num_return_sequences]]

                    # Save tmp file to debug
                    write_data(f"{save_dir}.tmp", {**new_obj, 'prediction_cots': prediction_cots})

                    answer_counter = Counter()
                    for i, prediction_value in enumerate(execute_fn(prediction_cots)):
                        if src_name == 'mathqa':
                            if len(prediction_value) != 1:
                                prediction_value = None
                        correctness = compare_answer_fn_mapper[src_name](prediction_value, target_value) if prediction_value is not None else False
                        new_obj["predictions"].append({
                            'completion': prediction_cots[i],
                            "solving_res": prediction_value,
                            "correctness": correctness,
                        })
                        if prediction_value is not None:
                            answer_counter[prediction_value] += 1

                    voting_answer = answer_counter.most_common(1)[0][0] if answer_counter else None
                    correctness = compare_answer_fn_mapper[src_name](voting_answer, target_value) if voting_answer is not None else False
                    acc += correctness
                    new_obj["most_voted_answer"] = voting_answer
                    new_obj["is_correct"] = correctness
                    all_results.append(new_obj)
                    
                write_data(f"{save_dir}.json", all_results)
                print(f"current_acc: {acc} / {len(all_results)} = {acc/len(all_results)* 100}%")
                if accelerator.is_main_process and args['wandb_log']:
                    wandb.log({'acc': acc/len(all_results)* 100}, step=b_idx)

    if accelerator.is_main_process:
        all_results = all_results[:len(tokenized_dataset)]
        write_data(f"{save_dir}.json", all_results)
        print(f"current_acc: {acc} / {len(all_results)} = {acc/len(all_results)* 100}%")
        if accelerator.is_main_process and args['wandb_log']:
            wandb.log({'acc': acc/len(all_results)* 100}, step=b_idx+1)


if __name__ == '__main__':
    from transformers import HfArgumentParser
    NONE_INT = -100 
    NONE_STR = 'None'
    @dataclass
    class Arguments:
        model_name: str
        input_path: str
        save_dir: str
        engine: str
        batch_size: int=field(default=2)
        max_length: int=field(default=1024)
        num_return_sequences: int=field(default=1)
        temperature: float=field(default=1.0)
        do_sample: bool=field(default=False)
        # wandb stuff
        wandb_log: bool = field(default=False)
        wandb_project: str = field(default='tmp_anvfupsadfn')
        wandb_run_name: str = field(default='default_run_name')

    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)
    for k,v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None
    print(args)
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))]) # wait for processing upto 5hrs
    main(args)
