'''
Pairwise evaluations v1

for example:
OPENAI_API_KEY=[key_here] python run_queries.py model_predictions/instruct_blip_eval_out.csv --engine gpt-3.5-turbo --task $task; done;

if --competing_csv is not passed, then the evaluation computed is win-rate vs. human.
if --competing_csv is passed, then the evaluation computed is model A vs. model B
'''
import argparse
from datasets import load_dataset
import pprint
import openai
import numpy as np
import json
import os
import time
import collections
import prompts
import copy
import tqdm
import pandas as pd

# Pairwise with reference (ref, cand A, cand B)
_PROMPT_SYSTEM_PAIRWISE_WITH_REFERENCE_V1, _PROMPT_USER_PAIRWISE_WITH_REFERENCE_V1, _PROMPT_ASSISTANT_PAIRWISE_WITH_REFERENCE_V1 = (
    prompts._PROMPT_SYSTEM_PAIRWISE_WITH_REFERENCE_V1,
    prompts._PROMPT_USER_PAIRWISE_WITH_REFERENCE_V1,
    prompts._PROMPT_ASSISTANT_PAIRWISE_WITH_REFERENCE_V1
)

# Pairwise (cand A [can be ref], cand B)
_PROMPT_SYSTEM_PAIRWISE_V1, _PROMPT_USER_PAIRWISE_V1, _PROMPT_ASSISTANT_PAIRWISE_V1  = (
    prompts._PROMPT_SYSTEM_PAIRWISE_V1,
    prompts._PROMPT_USER_PAIRWISE_V1,
    prompts._PROMPT_ASSISTANT_PAIRWISE_V1
)

# Multi-image
_PROMPT_SYSTEM_PAIRWISE_MULTI_IMAGE_V1, _PROMPT_USER_PAIRWISE_MULTI_IMAGE_V1, _PROMPT_ASSISTANT_PAIRWISE_MULTI_IMAGE_V1 = (
    prompts._PROMPT_SYSTEM_PAIRWISE_MULTI_IMAGE_V1,
    prompts._PROMPT_USER_PAIRWISE_MULTI_IMAGE_V1,
    prompts._PROMPT_ASSISTANT_PAIRWISE_MULTI_IMAGE_V1
)


# Answer extraction
_PROMPT_SYSTEM_ANSWER_EXTRACTION_V1, _ICL_SYSTEM_ANSWER_EXTRACTION_V1 = (
    prompts._PROMPT_SYSTEM_ANSWER_EXTRACTION_V1,
    prompts._ICL_SYSTEM_ANSWER_EXTRACTION_V1
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input_predictions_csv')
    parser.add_argument(
        '--competing_csv',
        default=None,
        type=str)

    parser.add_argument(
        '--limit',
        default=-1,
        type=int)

    parser.add_argument(
        '--engine',
        default='gpt-3.5-turbo',
        type=str)

    parser.add_argument(
        '--win_track_cache',
        default='head_to_heads.jsonl',
        type=str)


    args = parser.parse_args()

    #### START FROM HERE...
    args.cache = '{}_cache.jsonl'.format(args.engine)

    print('writing predictions to {}'.format(args.cache))

    return args


def generate_request(image_description,
                     instruction,
                     A, B, reference=None,
                     multi_image=False):
    global _PROMPT_SYSTEM_PAIRWISE_WITH_REFERENCE_V1, _PROMPT_USER_PAIRWISE_WITH_REFERENCE_V1, _PROMPT_ASSISTANT_PAIRWISE_WITH_REFERENCE_V1
    global _PROMPT_SYSTEM_PAIRWISE_V1, _PROMPT_USER_PAIRWISE_V1, _PROMPT_ASSISTANT_PAIRWISE_V1
    global _PROMPT_SYSTEM_PAIRWISE_MULTI_IMAGE_V1, _PROMPT_USER_PAIRWISE_MULTI_IMAGE_V1, _PROMPT_ASSISTANT_PAIRWISE_MULTI_IMAGE_V1

    if reference:
        system_p, user_p, assistant_p = _PROMPT_SYSTEM_PAIRWISE_WITH_REFERENCE_V1, _PROMPT_USER_PAIRWISE_WITH_REFERENCE_V1, _PROMPT_ASSISTANT_PAIRWISE_WITH_REFERENCE_V1
        user_content_message = ('OK. Here is the image description, the instruction, the high-quality reference, and the two response options, Response A and Response B.'
                                '\nImage context: {}\n\nInstruction:\n{}\n\nHigh quality reference: {}\n\nResponse A: {}\n\nResponse B: {}\n\nThink step-by-step and finish your response with "Overall, Response X is better." where X is either A or B.'.format(
                                    image_description, instruction, reference, A, B))
    elif multi_image:
        system_p, user_p, assistant_p = _PROMPT_SYSTEM_PAIRWISE_MULTI_IMAGE_V1, _PROMPT_USER_PAIRWISE_MULTI_IMAGE_V1, _PROMPT_ASSISTANT_PAIRWISE_MULTI_IMAGE_V1

        images_as_string = '\n'.join(['Image {}: {} '.format(idx+1, descr) for idx, descr in enumerate(image_description)])

        if reference is None:
            user_content_message = ('OK. Here is the sequence of image descriptions, the instruction, and the two response options, Response A and Response B.'
                                    '\nImage context:\n {}\n\nInstruction:\n{}\n\nResponse A: {}\n\nResponse B: {}\n\nThink step-by-step and finish your response with "Overall, Response X is better." where X is either A or B.'.format(
                                        images_as_string, instruction, A, B))
        else:
            user_content_message = ('OK. Here is the sequence of image descriptions, the instruction, and the two response options, Response A and Response B.'
                                    '\nImage context:\n {}\n\nInstruction:\n{}\n\nHigh quality reference: {}\n\nResponse A: {}\n\nResponse B: {}\n\nThink step-by-step and finish your response with "Overall, Response X is better." where X is either A or B.'.format(
                                        images_as_string, instruction, reference, A, B))
    else:
        system_p, user_p, assistant_p = _PROMPT_SYSTEM_PAIRWISE_V1, _PROMPT_USER_PAIRWISE_V1, _PROMPT_ASSISTANT_PAIRWISE_V1
        user_content_message = ('OK. Here is the image description, the instruction, and the two response options, Response A and Response B.'
                                '\nImage context: {}\n\nInstruction:\n{}\n\nResponse A: {}\n\nResponse B: {}\n\nThink step-by-step and finish your response with "Overall, Response X is better." where X is either A or B.'.format(
                                    image_description, instruction, A, B))
        
    messages = [{'role': 'system', 'content': system_p},
                {'role': 'user', 'content': user_p},
                {'role': 'assistant', 'content': assistant_p},
                {'role': 'user', 'content': user_content_message}]

    return messages



def generate_parsing_answer_request(query):
    global _PROMPT_SYSTEM_ANSWER_EXTRACTION_V1, _ICL_SYSTEM_ANSWER_EXTRACTION_V1

    messages = [{"role": "system", "content": _PROMPT_SYSTEM_ANSWER_EXTRACTION_V1}]

    for ex, res in _ICL_SYSTEM_ANSWER_EXTRACTION_V1:
        messages.append({
            'role': 'user',
            'content': ex + '\nPlease extract the final answer from the above text.'}
        )

        messages.append({
            'role': 'assistant',
            'content': res})

    messages.append({
        'role': 'user',
        'content': query + '\nPlease extract the final answer from the above text.'})

    return messages


def extract_prediction_from_response(resp, query2resp=None, cache=None):
    selected = {'{}'.format(ch): int('response {} is better'.format(ch.lower()) in resp.lower()) for ch in 'AB'}
    if np.sum(list(selected.values())) == 1:
        for k, v in selected.items():
            if v: return k
    
    else:
        messages = generate_parsing_answer_request(resp)
        query_as_key = messages[-1]['content']

        if query_as_key in query2resp:
            result = query2resp[query_as_key]
        else:
            api_result = None
            while api_result is None:
                try:
                    api_result = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo-0301',
                        messages=messages)
                except Exception as e:
                    print(e)
                    print('TIMEOUT. Sleeping and trying again.')
                    time.sleep(3)

            result = api_result['choices'][0]['message']['content']
            cache.write(json.dumps(
                {'query': query_as_key,
                 'response': result}) + '\n')

        selected = {'{}'.format(ch): int('Final Answer: Response {}'.format(ch) in result) for ch in 'AB'}
        if np.sum(list(selected.values())) == 1:
            for k, v in selected.items():
                if v:
                    return k
        return None


def judge(image_description,
          instruction,
          A, B,
          engine,
          query2resp,
          cache_f,
          reference=None,
          cached_only=False,
          multi_image=False):
    '''
    Returns "A" if A wins, "B" if B wins, or "Fail" if query failed.
    '''

    messages = generate_request(image_description, instruction, A, B, reference=reference, multi_image=multi_image)
    
    query_as_key = messages[-1]['content']

    if query_as_key in query2resp:
        result = query2resp[query_as_key]
    else:
        if cached_only:
            return None
        api_result = None

        while api_result is None:
            try:
                api_result = openai.ChatCompletion.create(
                    model=engine,
                    messages=messages)
            except Exception as e:
                print(e)
                print('TIMEOUT. Sleeping and trying again.')
                time.sleep(3)

        result = api_result['choices'][0]['message']['content']
        cache_f.write(json.dumps(
            {'query': query_as_key,
             'response': result}) + '\n')

    prediction = extract_prediction_from_response(result, query2resp=query2resp, cache=cache_f)
    return prediction, result, messages
