import argparse
from datasets import load_dataset
import pprint
import openai
import numpy as np
import json
import os
import collections
import glob
import tqdm
import pandas as pd
import pairwise_eval

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
        '--extra_cache_filename',
        default='',
        type=str)

    parser.add_argument(
        '--multi_image',
        default=0,
        type=int)

    parser.add_argument(
        '--leaderboard_jsonl',
        required=True,
        type=str)

    args = parser.parse_args()

    print('args:')
    print(args)

    args.cache = '{}_cache{}.jsonl'.format(args.engine, args.extra_cache_filename)

    print('writing predictions to {}'.format(args.cache))

    return args


def main():
    args = parse_args()
    np.random.seed(2)

    instances = []
    with open(args.leaderboard_jsonl) as f:
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
    np.random.shuffle(instances)
    bar = tqdm.tqdm(instances)

    for inst in bar:

        pairwise_eval.judge(
            inst['image_dense_caption'],
            inst['instruction'],
            A=inst['A'],
            B=inst['B'],
            query2resp=query2resp,
            engine=args.engine,
            cache_f=cache,
            multi_image=args.multi_image)

        pairwise_eval.judge(
            inst['image_dense_caption'],
            inst['instruction'],
            A=inst['B'],
            B=inst['A'],
            query2resp=query2resp,
            engine=args.engine,
            cache_f=cache,
            multi_image=args.multi_image)


if __name__ == '__main__':
    main()
