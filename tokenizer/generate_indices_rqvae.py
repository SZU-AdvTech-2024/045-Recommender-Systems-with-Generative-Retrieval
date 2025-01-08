import collections
import json
import os.path

import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
import argparse

from model.rqvae import RQVAE
from utils import ensure_dir, TokenizerDataset


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--dataset', type=str, default='Games')
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--K', type=int, default=256)
    parser.add_argument('--D', type=int, default=3)
    parser.add_argument('--ckpt_dir', type=str, default='./saved_model')
    parser.add_argument('--tokenizer_plm', type=str, default='sentence-t5-base')
    parser.add_argument('--ckpt_file', type=str, default='best_collision_model.pth')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--token_type', type=str, default='pretrained', choices=['sid', 'pretrained'])

    args = parser.parse_args()

    # ckpt: checkpoint
    ckpt_dir=os.path.join(args.ckpt_dir, f'{args.token_type}')
    if args.token_type == 'sid':
        dataset_name = f'{args.dataset}_{args.tokenizer_plm}'
    elif args.token_type == 'pretrained':
        dataset_name = f'{args.dataset}'
    ckpt_dir = os.path.join(ckpt_dir, dataset_name)
    token_size = f'K={args.K}_D={args.D}'
    ckpt_dir = os.path.join(ckpt_dir, token_size)
    ckpt_path = os.path.join(ckpt_dir, args.ckpt_file)

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    # setting output dir
    output_dir = f"./{args.token_type}_result"
    output_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(output_dir)
    output_file = token_size + '_index.json'
    output_path = os.path.join(output_dir, output_file)

    # load item embed
    data_dir = os.path.join(args.data_dir, args.dataset)
    if args.token_type == 'sid':
        file_name=f'text_embed_{args.tokenizer_plm}.pt'
    elif args.token_type == 'pretrained':
        file_name='pretrained_sasrec.pt'
    data_path = os.path.join(data_dir, file_name)

    semantic_embed = torch.load(data_path)
    data = TokenizerDataset(semantic_embed)

    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

    model = RQVAE(in_dim=semantic_embed.shape[-1],
                  num_emb_list=model_args.num_emb_list,
                  e_dim=model_args.e_dim,
                  layers=model_args.layers,
                  dropout_prob=model_args.dropout_prob,
                  bn=model_args.bn,
                  loss_type=model_args.loss_type,
                  quant_loss_weight=model_args.quant_loss_weight,
                  kmeans_init=model_args.kmeans_init,
                  kmeans_iters=model_args.kmeans_iters,
                  )

    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    print(model)


    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>"]

    for d in tqdm(data_loader):
        d = d.to(args.device)
        indices = model.get_indices(d)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices.append(code)
            all_indices_str.append(str(code))
        # break

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    collision_rate = (tot_item - tot_indice) / tot_item
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    with open(output_path, 'w') as fp:
        json.dump(all_indices_dict, fp)
