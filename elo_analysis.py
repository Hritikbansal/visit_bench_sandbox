import argparse
from collections import defaultdict
import datetime
import json
import math
import pickle
from pytz import timezone
import numpy as np
import pandas as pd
from tqdm import tqdm
import collections
import datetime

pd.options.display.float_format = "{:.2f}".format

def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    # battles is a list of (A, B, who won or "tie")
    rating = defaultdict(lambda: INIT_RATING)

    for model_a, model_b, win in battles:
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if win == "model_a":
            sa = 1
        elif win == "model_b":
            sa = 0
        elif win == "tie" or win == "tie (bothbad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {win}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    return dict(rating)


def get_bootstrap_result(battles, func_compute_elo, num_round=1000):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        tmp_battles = battles.sample(frac=1.0, replace=True)
        # tmp_battles = tmp_battles.sort_values(ascending=True, by=["tstamp"])
        rows.append(func_compute_elo(tmp_battles))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def get_elo_from_bootstrap(bootstrap_df):
    return dict(bootstrap_df.quantile(0.5))


def compute_pairwise_win_fraction(battles, model_order):
    # Times each model wins as Model A
    a_win_ptbl = pd.pivot_table(
        battles[battles["win"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting times each model wins as Model B
    b_win_ptbl = pd.pivot_table(
        battles[battles["win"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting number of A-B pairs
    num_battles_ptbl = pd.pivot_table(
        battles, index="model_a", columns="model_b", aggfunc="size", fill_value=0
    )

    # Computing the proportion of wins for each model as A and as B
    # against all other models
    row_beats_col_freq = (a_win_ptbl + b_win_ptbl.T) / (
        num_battles_ptbl + num_battles_ptbl.T
    )

    if model_order is None:
        prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
        model_order = list(prop_wins.keys())

    # Arrange ordering according to proprition of wins
    row_beats_col = row_beats_col_freq.loc[model_order, model_order]
    return row_beats_col


def pretty_print_elo_rating(rating, battle_counts):
    model_order = list(rating.keys())
    model_order.sort(key=lambda k: -rating[k])
    print(f"ranking, model, elo, battles")
    for i, model in enumerate(model_order):
        print(f"{i+1:2d}, {model:25s}, {rating[model]:.0f}, {battle_counts[model]}")

def get_elo_as_tsv(rating, pairs2win, battle_counts, args):
    model_order = list(rating.keys())
    model_order.sort(key=lambda k: -rating[k])
    data = []

    model2wrvr = {'human_verified_reference': '---'}
    for p, win in pairs2win.items():
        if 'human_verified_reference' in p:
            non_human = [x for x in p if x != 'human_verified_reference'][0]
            model2wrvr[non_human] = '{:.2f}%'.format(100* (collections.Counter(win)[non_human] / len(win))) + ' (n={})'.format(len(win))

    for i, model in enumerate(model_order):
        data.append(
            {'Category':args.tsv_category,
             'Model': str(model),
             'Elo': '{:.0f}'.format(rating[model]),
             '# Matches': str(battle_counts[model]),
             'Win vs. Reference (w/ # ratings)': model2wrvr[model] if model in model2wrvr else 'TBD! RUN MORE',
            })

    data = pd.DataFrame(data)
    # Get today's date
    today = datetime.date.today()

    # Format the date as "MonDDYYYY"
    formatted_date = today.strftime("%b%d%Y")

    fname = 'visitbench_leaderboard_{}_{}.tsv'.format(args.tsv_category.replace(' ', '~'), formatted_date)
    print('writing current leaderboard to {}'.format(fname))
    data.to_csv(fname, sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--head2head_file", type=str, default='gpt-4_head2head.json')
    parser.add_argument("--tsv_category", type=str, default='Single Image')
    args = parser.parse_args()

    battles = []
    pairs2win = collections.defaultdict(list)

    battle_count = collections.Counter()

    model2instance2count = collections.defaultdict(lambda : collections.Counter())
        
    with open(args.head2head_file) as f:
        all_results = json.load(f)
        done = set()

        for results in all_results:
            k = tuple(sorted(list([results['A_model'], results['B_model']])))
            done_check_k = (k, results['image_url'], results['instruction'])
            if done_check_k in done:
                continue
            done.add(done_check_k)
            battle_count[k] += 1

            model2instance2count[results['A_model']][(results['image_url'], results['instruction'])] += 1
            model2instance2count[results['B_model']][(results['image_url'], results['instruction'])] += 1

            wins = [y for y in results['auto_evaluation_result'] if y != 'tie']
            if len(wins) == 0 or len(wins) == len(set(wins)): # true tie.
                # random choice for comparison ...
                results['winner'] = 'tie'
            else:
                winner = collections.Counter(wins).most_common(1)[0][0]
                results['winner'] = 'A' if winner == results['A_model'] else 'B'
                pairs2win[k].append(collections.Counter(wins).most_common(1)[0][0])

            if results['winner'] == 'A':
                res_str = 'model_a'
            elif results['winner'] == 'B':
                res_str = 'model_b'
            elif results['winner'] == 'tie':
                res_str = 'tie'
            else:
                print(results['winner'])
                print('wtf')
                quit()
            battles.append((results['A_model'], results['B_model'], res_str))

    for m, inst2count in model2instance2count.items():
        print(m)
        print(sorted(inst2count.values(), key=lambda x: -x))

            
    print('Win rate versus reference, majority vote:')
    for p, win in pairs2win.items():
        if 'human_verified_reference' in p:
            non_human = [x for x in p if x != 'human_verified_reference'][0]
            print(non_human, '{:.2f}%'.format(100* (collections.Counter(win)[non_human] / len(win))), 'n={}'.format(len(win)))

    np.random.seed(1)
    np.random.shuffle(battles)
    model_match_count = collections.Counter()
    for m1, m2, _ in battles:
        model_match_count[m1] += 1
        model_match_count[m2] += 1

    print(model_match_count)
    print(len(battles))
    elos = compute_elo(battles)
    pretty_print_elo_rating(elos, model_match_count)
    get_elo_as_tsv(elos, pairs2win, model_match_count, args)
