#!/usr/bin/env python
# coding: utf-8

import os
import torch
import json
from huggingface_hub import login
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import pipeline
from vllm import LLM, SamplingParams

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
# use export CUDA_VISIBLE_DEVICES if running sbatch

def init_llama(num_gpus, model_dir):
    """
    Uses vllm's interface to load large models over multiple GPU's
    """
    #login('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')
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model="meta-llama/Llama-2-13b-chat-hf",\
              tensor_parallel_size=num_gpus,\
              download_dir=model_dir, \
              tokenizer='hf-internal-testing/llama-tokenizer')
    return llm, sampling_params




def generate_prompt(prompt):
    sys_prompt =  prompt['generation_prompt']['system'] 
    user_prompt = prompt['generation_prompt']['user']
    gen_prompt = "<s>[INST] <<SYS>>\n" + sys_prompt + "\n<</SYS>>" + "\n\n" + user_prompt + " [/INST]"
    return gen_prompt 



def run_llama(model_dir, outpath, num_gpus, batch_size):
    llm, sampling_params = init_llama(num_gpus, model_dir)
    prompts = json.load(open("/people/spru445/json_database.json", "r"))
    #print(prompts)
    for i in range(0, len(prompts), batch_size):
        print("running prompts ", i, i+batch_size)
        batch_prompts = [generate_prompt(x) for x in prompts[i:i+batch_size]]
        #print(batch_prompts)
        answers = llm.generate(batch_prompts, sampling_params)
        for output in answers:
            generated_text = output.outputs[0].text
            prompts[i]['generation_prompt']['llama_answer'] = generated_text
    with open(outpath, "w") as outfile:
        json.dump(prompts, outfile)



if __name__ == "__main__":
    #test_llama(num_gpus=8, model_dir='/qfs/projects/va_aprx/')
    run_llama(outpath='llama_output.json', num_gpus=2,\
    model_dir='/qfs/projects/va_aprx/', batch_size=4)
