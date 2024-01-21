'''
Runs all queries in a given input jsonlist, e.g.,

python evaluate_all_queries.py
'''
import argparse
from datasets import load_dataset
import pprint
import openai
import numpy as np
import json
import os
import collections
import tqdm
import pandas as pd
import pairwise_eval
import glob
import datetime


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--limit',
        default=-1,
        type=int)

    parser.add_argument(
        '--engine',
        default='gpt-4',
        type=str)

    parser.add_argument(
        '--multi_image',
        default=0,
        type=int)

    args = parser.parse_args()

    args.cache = '{}_cache.jsonl'.format(args.engine)
    args.output_f = '{}_head2head.json'.format(args.engine)

    print('reading predictions from {}'.format(args.cache))
    print('outputting ref-free from {}'.format(args.output_f))

    return args


def main():
    args = parse_args()
    np.random.seed(2)

    leaderboard_jsonls = glob.glob('leaderboard_submission_model_queries/*.jsonl') + ['all_pairs_with_references_original_set.jsonl']

    instances = []
    for leaderboard_jsonl in leaderboard_jsonls:
        with open(leaderboard_jsonl) as f:
            for line in f:
                instances.append(json.loads(line.strip()))

    query2resp = {}
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            for line in f:
                d = json.loads(line)
                query = d['query']
                resp = d['response']
                query2resp[query] = resp
    print('Loaded {} queries'.format(len(query2resp)))

    cache = open(args.cache, 'a')

    bar = tqdm.tqdm(instances)

    results = []
    reference_results = []

    for inst in bar:
        try:
            winner_1, cot_1, _ = pairwise_eval.judge(
                inst['image_dense_caption'],
                inst['instruction'],
                A=inst['A'],
                B=inst['B'],
                query2resp=query2resp,
                engine=args.engine,
                cache_f=cache,
                cached_only=True,
                multi_image=args.multi_image)
            if winner_1:
                winner_1 = inst['A_model'] if winner_1 == 'A' else inst['B_model']
            else:
                winner_1 = 'tie'

        except:
            continue
        try:
            winner_2, cot_2, _ = pairwise_eval.judge(
                inst['image_dense_caption'],
                inst['instruction'],
                A=inst['B'],
                B=inst['A'],
                query2resp=query2resp,
                engine=args.engine,
                cache_f=cache,
                cached_only=True,
                multi_image=args.multi_image)
            if winner_2:
                winner_2 = inst['A_model'] if winner_2 == 'B' else inst['B_model']
            else:
                winner_2 = 'tie'
        except:
            continue

        results.append({'A_model': inst['A_model'],
                        'B_model': inst['B_model'],
                        'A': inst['A'],
                        'B': inst['B'],
                        'auto_evaluation_result': [winner_1, winner_2],
                        'auto_evaluation_cot': [
                            {'A_model_in_cot': inst['A_model'],
                             'B_model_in_cot': inst['B_model'],
                             'cot': cot_1},
                            {'A_model_in_cot': inst['B_model'],
                             'B_model_in_cot': inst['A_model'],
                             'cot': cot_2},],
                        'image_url': inst['image_url'] if 'image_url' in inst else inst['image_urls'],
                        'instruction': inst['instruction'],
                        'instruction_category': inst['instruction_category'],
                        'engine': args.engine,
                        'reference': inst['reference'],
                        'evaluated_with_reference': False})

    print('writing {} head-to-head results to {}'.format(len(results), args.output_f))
    with open(args.output_f, 'w') as f:
        f.write(json.dumps(results))


if __name__ == '__main__':
    main()
