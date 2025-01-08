import argparse
import json
from collections import defaultdict
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Games')
    parser.add_argument('--tokenizer_plm', type=str, default='sentence-t5-base',
                        choices=['sentence-t5-base', 'bert-base-uncased'])
    parser.add_argument('--K', type=int, default=256)
    parser.add_argument('--D', type=int, default=3)
    # cid has not conflict
    parser.add_argument('--token_type', type=str, default='pretrained', choices=['sid', 'pretrained'])

    args = parser.parse_args()

    indices_dir = f'./{args.token_type}_result'
    if args.token_type == 'sid':
        dataset_name = args.dataset + '_' + args.tokenizer_plm
    else:
        dataset_name = args.dataset
    indices_dir = os.path.join(indices_dir, dataset_name)
    file_name = f'K={args.K}_D={args.D}_index.json'
    indices_path = os.path.join(indices_dir, file_name)

    with open(indices_path, 'r') as f:
        id2indices = json.load(f)

    id_indices_dict = defaultdict(list)
    for item_id, indices in id2indices.items():
        item_indices = ''.join(indices)
        id_indices_dict[item_indices].append(item_id)

    no_conflict_index = dict()
    template = f'<{chr(ord("a") + args.D)}_{{}}>'
    for indices, item_id_list in id_indices_dict.items():
        cnt = 0
        for item_id in item_id_list:
            nc_indices = indices + template.format(cnt)

            nc_indices_list = []
            left = 0
            for right in range(len(nc_indices)):
                if nc_indices[right] == '>':
                    nc_indices_list.append(nc_indices[left:right + 1])
                    left = right + 1

            # 转换成多个列表形式
            no_conflict_index[item_id] = nc_indices_list

            cnt += 1

    save_file = f'K={args.K}_D={args.D}_index_nc.json'
    save_path = os.path.join(indices_dir, save_file)
    with open(save_path, 'w') as f:
        json.dump(no_conflict_index, f)
