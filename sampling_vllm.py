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
    assert src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric', 'math']
    global instruction
    global cot_trigger
    global answer_trigger
    # Complete output is in this form: f'{instruction}{question.strip()}{cot_trigger}{answer_cot.strip()}'
    instruction = 'Question:\n'
    cot_trigger = '\nAnswer reasoning:\n'
    answer_trigger = '\nTherefore, the answer is: '
    return 

post_process_final_answer_fn_mapper = {
    ('python', 'gsm8k'): lambda question, x: float(x.replace(',','').strip()),
    ('python', 'svamp'): lambda question, x: float(x.replace(',','').strip()),
    ('python', 'mathqa'): lambda question, x: x.lower().replace('"','').replace("'",'').strip(),
    ('python', 'mathqa-numeric'): lambda question, x: float(x),
    ('nl', 'gsm8k'): lambda question, x: float(x.replace(',','').strip()),
    ('nl', 'svamp'): lambda question, x: float(x.replace(',','').strip()),
    ('nl', 'mathqa'): lambda question, x: x.lower().replace('"','').replace("'",'').strip(),
    ('nl', 'mathqa-numeric'): lambda question, x: float(x),
}
### the answer_cot is a list of answer_cot
post_process_answer_cot_fn_mapper = {
    ('python', 'gsm8k'): lambda question, answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'svamp'): lambda question, answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'mathqa'): lambda question, answer_cot: [str(res).lower().replace('"','').replace("'",'').strip() for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('python', 'mathqa-numeric'): lambda question, answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    ('nl', 'gsm8k'): lambda question, answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    ('nl', 'svamp'): lambda question, answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    ('nl', 'mathqa'): lambda question, answer_cot: [res.split(answer_trigger)[-1].lower().replace('"','').replace("'",'').strip() for res in answer_cot],
    ('nl', 'mathqa-numeric'): lambda question, answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
}
compare_answer_fn_mapper = {
    ('python', 'gsm8k'): lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    ('python', 'svamp'): lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    ('python', 'mathqa'): lambda extracted_ans, target_answer: extracted_ans == target_answer,
    ('python', 'mathqa-numeric'): lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    ('nl', 'gsm8k'): lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    ('nl', 'svamp'): lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    ('nl', 'mathqa'): lambda extracted_ans, target_answer: extracted_ans == target_answer,
    ('nl', 'mathqa-numeric'): lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
}

def tokenize_fn(examples: Dict, tokenizer, max_length, src_name, engine, model_name_or_path):
    features = {"input_ids": [], "attention_mask": [], "answer_value": [], "answer_cot": [], "question": [], 'item_id': []}
    for idx, question in enumerate(examples["question"]):
        text = f"{instruction}{question}{cot_trigger}"
        if src_name in ['gsm8k', 'mathqa', 'svamp'] and engine == 'python':
            text += f'def solution():\n    """{question}"""'
        source_text_res = tokenizer.encode_plus(text, max_length=max_length, truncation=True, add_special_tokens=True)
        features["input_ids"].append(source_text_res["input_ids"])
        features["attention_mask"].append(source_text_res["attention_mask"])

        if 'gemma' in model_name_or_path:
            # Quick fix for gemma -- https://github.com/huggingface/transformers/issues/29250
            features["input_ids"][-1] = [tokenizer.bos_token_id] + features["input_ids"][-1]
            features["attention_mask"][-1] = [1] + features["attention_mask"][-1]

        features["question"].append(question)
        features["answer_value"].append(examples["answer_value"][idx])
        features["answer_cot"].append(None if "answer_cot" not in examples else examples["answer_cot"][idx])
        features['item_id'].append(examples['item_id'][idx])
    return features

def collate_fn(features, tokenizer):
    return {'input_ids': [feat['input_ids'] for feat in features], 'attention_mask': [feat['attention_mask'] for feat in features]}

def main(args):
    set_seed(42)
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
    if tokenizer.eos_token_id is None: 
        tokenizer.add_special_tokens({'eos_token': '<eos>'})
    if tokenizer.pad_token_id is None or (tokenizer.eos_token_id == tokenizer.pad_token_id):
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    # loading training data
    raw_dataset = Dataset.from_list(json.load(open(input_path,'r')))
    print('Raw data:', raw_dataset)
    src_name = raw_dataset[0]['item_id'].split('_')[0]  # e.g., gsm8k_0, gsm8k_1, gsm8k_2, ...
    setup_cot(src_name)
    print('Using instruction:', instruction)
    print('Using cot_trigger:', cot_trigger)
    print('Using answer_trigger:', answer_trigger)
    tokenized_dataset = raw_dataset.map(
        tokenize_fn, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'src_name': src_name, 'engine': engine, 'model_name_or_path': model_name},
        batched=True, remove_columns=raw_dataset.column_names, load_from_cache_file=False, num_proc=8
    )
    valid_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=partial(collate_fn, tokenizer=tokenizer))

    # loading model
    from vllm import LLM, SamplingParams
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    model = LLM(model=model_name, tensor_parallel_size=8, dtype=torch.bfloat16, enforce_eager=True)
    sampling_params = SamplingParams(n=num_return_sequences, temperature=temperature if do_sample else 0, max_tokens=max_length)
    all_results = []
    acc = 0
    for b_idx, batch in (pbar:= tqdm(enumerate(valid_dataloader), desc="Generating", total=len(valid_dataloader))):
        with torch.no_grad():
            outputs = model.generate(prompt_token_ids=batch["input_ids"], sampling_params=sampling_params, use_tqdm=False)
            outputs = [output.prompt_token_ids + o.token_ids for output in outputs for o in output.outputs]
            cur_batch_size = len(outputs)//num_return_sequences
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
                execute_fn = post_process_answer_cot_fn_mapper[(engine, src_name)]
                target_value = post_process_final_answer_fn_mapper[(engine, src_name)](obj['question'], obj['answer_value'])
                prediction_cots = [pred.strip().split(cot_trigger)[-1] for pred in decoded_output[b_item_idx*num_return_sequences: (b_item_idx+1)*num_return_sequences]]
                # Save tmp file to debug
                write_data(f"{save_dir}.tmp", {**new_obj, 'prediction_cots': prediction_cots})
                answer_counter = Counter()
                for i, prediction_value in enumerate(execute_fn([obj["question"]]*len(prediction_cots), prediction_cots)):
                    if src_name == 'mathqa':
                        if len(prediction_value) != 1:
                            prediction_value = None
                    correctness = compare_answer_fn_mapper[(engine,src_name)](prediction_value, target_value) if prediction_value is not None else False
                    new_obj["predictions"].append({
                        'completion': prediction_cots[i],
                        "solving_res": prediction_value,
                        "correctness": correctness,
                    })
                    if prediction_value is not None:
                        answer_counter[prediction_value] += 1

                voting_answer = answer_counter.most_common(1)[0][0] if answer_counter else None
                correctness = compare_answer_fn_mapper[(engine, src_name)](voting_answer, target_value) if voting_answer is not None else False
                acc += correctness
                new_obj["most_voted_answer"] = voting_answer
                new_obj["is_correct"] = correctness
                all_results.append(new_obj)
                
            write_data(f"{save_dir}.json", all_results)
            print(f"current_acc: {acc} / {len(all_results)} = {acc/len(all_results)* 100}%")
            # pbar.set_postfix({'current_acc': f'{acc} / {len(all_results)} = {acc/len(all_results)* 100}%'})

    all_results = all_results[:len(tokenized_dataset)]
    write_data(f"{save_dir}.json", all_results)
    print(f"current_acc: {acc} / {len(all_results)} = {acc/len(all_results)* 100}%")

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
        do_sample: bool=field(default=False)
        temperature: float=field(default=1)
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
    main(args)
