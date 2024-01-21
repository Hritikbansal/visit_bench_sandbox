'''
Samples a random set of pairs for human correlation with GPT-4 Eval style.
'''
import argparse
import pandas as pd
import pprint
import glob
import collections
import itertools
import numpy as np
import json
import os

# These are the ones for which we will do all pairs. All others will have a random set of pairs generated.
_original_model_csvs = [
    'full_model_predictions/otter_single_fixed.csv',
    'full_model_predictions/llamaadapter-v2_7b.csv',
    'full_model_predictions/instruct_blip_13b_out.csv',
    'full_model_predictions/visualgpt_davinci003_out.csv',
    'full_model_predictions/PandaGPT_13b_out.csv',
    'full_model_predictions/llava13b_output.csv',
    'full_model_predictions/minigpt-4_7b.csv',
    'full_model_predictions/openflamingo_single_fixed.csv',
    'full_model_predictions/mmgpt_7b_output_column.csv',
    'full_model_predictions/mplug-owl_7b.csv'
]
_original_model_names = set(['human_verified_reference'])
_leaderboard_model_names = set()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_reference_comparisons',type = int, default = 150, help = 'number of comparisons with the reference output')
    return parser.parse_args()


def main():
    global _original_model_csvs, _original_model_names
    np.random.seed(1)
    args = parse_args()
    dataset = pd.read_csv('single_image_full_dataset.csv').to_dict(orient='records')
    final_set = []
    print('starting with {} instances'.format(len(dataset)))
    for d in dataset:
        if d['human_ratings_gpt4_correct'] == 1 and d['human_ratings_problem_in_caption'] == 0:
            final_set.append(d)

    print('final set with correct gpt4 and no human ratings problem in caption: {}'.format(len(final_set)))

    # maps from (url, instruction) --> [{'model': ..., 'response': ...}]
    predictions = collections.defaultdict(list)
    references = {}
    image_dense_caption = {}

    for d in final_set:
        k = (d['image'], d['instruction'], d['instruction_category'])
        predictions[k].append({'model': 'human_verified_reference', 'response': d['gpt4_prediction']})
        references[k] = d['gpt4_prediction']
        image_dense_caption[d['image']] = d['image_dense_captions']


    predictions_csvs = glob.glob('full_model_predictions/*.csv')

    for p in predictions_csvs:
        
        preds = pd.read_csv(p).to_dict(orient='records')
        ks = preds[0].keys()
        k1 = [k for k in ks if k not in set(['image_url', 'image_dense_caption', 'instruction_category', 'instruction',
                                             'output_human_annotated', 'image', 'gpt4_prediction', 'image_dense_captions',
                                             'reference_output', 'human_ratings_gpt4_correct', 'human_ratings_problem_in_caption',
                                             'human_ratings_problem_in_gpt4','public_images_metadata', 'Unnamed', 'visual',
                                             'Unnamed: 0', 'idx', 'question','image_id'])]
        print(p, len(preds))
        assert len(k1) == 1, k1
        k1 = k1[0]

        if p in _original_model_csvs:
            _original_model_names.add(k1)
        else:
            _leaderboard_model_names.add(k1)
        
        for p in preds:
            if (p['image'], p['instruction'], p['instruction_category']) in predictions: # skip cases that dont meet initial filter
                predictions[(p['image'], p['instruction'], p['instruction_category'])].append({'model': k1, 'response': p[k1]})


    # all possible pairs from original set
    all_pairs = []
    new_leaderboard_pairs = []
    for k, v in predictions.items():
        for r1, r2 in itertools.combinations(v, 2):

            if r1['model'] not in _original_model_names or r2['model'] not in _original_model_names:
                lst = new_leaderboard_pairs
            else:
                lst = all_pairs
            if np.random.random() < .5:
                A, B = r1, r2
            else:
                A, B = r2, r1
                
            if A['model'] == B['model']: continue
                
            lst.append({'image_url': k[0],
                        'instruction': k[1],
                        'instruction_category': k[2],
                        'A': A['response'],
                        'B': B['response'],
                        'A_model': A['model'],
                        'B_model': B['model']})

    np.random.shuffle(all_pairs)

    for p in all_pairs + new_leaderboard_pairs:
        p['reference'] = references[(p['image_url'], p['instruction'], p['instruction_category'])]
        p['image_dense_caption'] = image_dense_caption[p['image_url']]

    with open('all_pairs_with_references_original_set.jsonl', 'w') as f:
        f.write('\n'.join([json.dumps(p) for p in all_pairs]))

    # OK, now we loop over new models, and generate their new pairs files if they don't already exist.
    if not os.path.exists('leaderboard_submission_model_queries'):
        os.makedirs('leaderboard_submission_model_queries')
    
    for leaderboard_model_name in _leaderboard_model_names:
        print(leaderboard_model_name)
        new_fn = 'leaderboard_submission_model_queries/{}_new_pairs_with_references.jsonl'.format(leaderboard_model_name.replace(' ', '~'))
        if os.path.exists(new_fn): continue
        
        # get pairs to query for the new model
        print('writing new pairs to query to do leaderboarding for {} to {}'.format(leaderboard_model_name, new_fn))
        instance2valid_pairs = collections.defaultdict(list)
        instance2reference_compare = {}
        for p in new_leaderboard_pairs:
            k = (p['image_url'], p['instruction'], p['instruction_category'])
            if leaderboard_model_name in [p['A_model'], p['B_model']]:
                if 'human_verified_reference' in [p['A_model'], p['B_model']]:
                    instance2reference_compare[k] = p
                else:
                    instance2valid_pairs[k].append(p)
        
        # seed based on the name of the input model
        np.random.seed(hash(leaderboard_model_name)%(2**32 - 1))

        # reference compares
        new_to_run = list(instance2reference_compare.values())
        np.random.shuffle(new_to_run)
        new_to_run = new_to_run[:args.num_reference_comparisons]

        # one random other model
        for k, v in instance2valid_pairs.items():
            new_to_run.append(np.random.choice(v))
        print('{} new pairwise comparisons for {}'.format(len(new_to_run), leaderboard_model_name))

        np.random.shuffle(new_to_run)
        with open(new_fn, 'w') as f:
            f.write('\n'.join([json.dumps(p) for p in new_to_run]))

if __name__ == '__main__':
    main()
