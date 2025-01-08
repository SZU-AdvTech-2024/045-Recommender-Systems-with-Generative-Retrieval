import argparse
import json

import pandas as pd
from tqdm import tqdm


def print_data_info(_args, _data):
    user_num, item_num = _data['user'].max() + 1, data['item'].max() + 1
    user_inters = _data['user'].value_counts()
    seq_avg_len = user_inters.mean()
    print(f'Dataset: {args.dataset}')
    print(f'user num: {user_num}, item_num: {item_num}, sequence average length: {round(seq_avg_len, 2)}')

    # ensure no cold user
    assert user_inters.min() >= 5


# leave-one-out for temporary
# split by timestamp to be finished
def split_seq(_data, _args):
    grouped = _data.groupby(by=['user'])

    # process
    inter_data=dict()
    for user, seq in tqdm(grouped):
        sorted_seq = seq.sort_values(by='timestamp')['item'].values.tolist()
        user = user[0]
        inter_data[user] = sorted_seq

    # write
    with open(f'{_args.dataset}/inter_data.json', 'w') as fp:
        json.dump(inter_data, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Bili_Cartoon')
    parser.add_argument('--max_len', type=int, default=20)

    args = parser.parse_args()

    data = pd.read_csv(f'./{args.dataset}/{args.dataset}_pair.csv', names=['item', 'user', 'timestamp'])
    print_data_info(args, data)

    split_seq(data, args)
