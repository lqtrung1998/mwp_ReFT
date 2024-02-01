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
from typing import Any, Dict
from src.utils import timeout
import time
from tqdm import tqdm
import numpy as np
from pebble import ProcessPool
import sympy
import math
import copy
global_restricted = {lib: globals()[lib] for lib in ['sympy', 'math']}
# del global_restricted['sympy'].init_session
local_restricted = {}

def exec_code(code_piece, _global_vars, _local_vars):
    exec(code_piece, _global_vars, _local_vars)
    
def eval_code(expr, _global_vars, _local_vars):
    return eval(expr, _global_vars, _local_vars)

def run(code_piece, expr):
    _global_vars, _local_vars = {}, {}
    for lib in ['sympy', 'math']:
        _global_vars[lib] = global_restricted[lib]
        if lib in local_restricted:
            _local_vars[lib] = local_restricted[lib]
    exec(code_piece, _global_vars, _local_vars)
    result = eval(expr, _global_vars, _local_vars)
    return result

def process_code(code_gen, truncate_first_return=False):
    ## deal with blacklist keyword
    if 'sys.exit' in code_gen:
        code_gen = code_gen.replace('sys.exit', 'print')
    snippet = code_gen.split('\n')
    ## post process the code
    updated_code_snippet = ['import math', 'import sympy']
    for snippet_line in snippet:
        if snippet_line.startswith('def solution'):
            updated_code_snippet.append(snippet_line)
            continue
        if snippet_line.strip() == "":
            break
        if truncate_first_return:
            if snippet_line == "    return result":
                break
        updated_code_snippet.append(snippet_line)
    updated_code_gen = '\n'.join(updated_code_snippet)
    return updated_code_gen

def run_python_code(programs, TIMEOUT: float, safe=True):
    is_single_program = False
    if not isinstance(programs, list):
        is_single_program = True
        programs = [programs]
    updated_programs = [process_code(code) for code in programs]
    if safe:
        # Safer -- executed code can't affect main code (e.g numpy.random.seed(...))
        # But it is slow ... 
        with ProcessPool(max_workers=8) as pool:
            futures = [pool.schedule(run, args=[code,  'solution()'], timeout=TIMEOUT) for code in updated_programs]
            results = []
            for i, f in tqdm(enumerate(futures), total=len(futures), disable=True):
                try:
                    res = f.result()
                except Exception as e:
                    print(str(e)) #, updated_programs[i])
                    res = None
                results.append(res)
    else:
        results = []
        for code in tqdm(updated_programs, disable=True):
            with timeout(seconds=int(TIMEOUT)):
                try:
                    res = run(code_piece=code, expr="solution()")
                except Exception as e:
                    print(str(e), code)
                    res = None
                results.append(res)

    if is_single_program:
        assert len(results) == 1, len(results)
        return results[0]

    return results


if __name__ == '__main__': 
    code = '''
    def solution():
        """Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""
        import time
        time.sleep(2)
        from sympy import init_session
        init_session()
        # raise
        clips_april = 48
        clips_may = clips_april / 2
        clips_total = clips_april + clips_may
        result = clips_total
        # import numpy as np
        # np.random.seed(42)
        # return np.random.randint(10)
        # np.random.seed(42)
        return result
    '''.strip()
    print(code)
    s = time.time()
    for i in tqdm(range(1)):
        res = run_python_code([code]*10, 2.5, safe=True)
        print(res)
    print(time.time()-s)
    print(np.random.randint(10))
    sum([elem for elem in res if elem is not None])/len(res)
