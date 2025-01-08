import argparse
import logging
from collections import defaultdict

import torch
import os
import math

import json

from utils import set_seed, RecDataset, RecCollator, ensure_dir, get_local_time, Trie, prefix_allowed_tokens_fn, \
    setup_logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = 'false'


# metric function
def ndcg_at_k(_rec_list, k):
    ndcg = 0.0

    for one_rec in _rec_list:
        topk_rec = one_rec[:k]
        for i in range(len(topk_rec)):
            ndcg += topk_rec[i] / math.log(i + 2, 2)

    return ndcg


def hit_at_k(_rec_list, k):
    hit = 0.0

    for one_rec in _rec_list:
        topk_rec = one_rec[:k]
        if sum(topk_rec) > 0:
            hit += 1

    return hit


# get metrics result
def get_metrics(metrics, _rec_list):
    _metrics_dict = dict()

    metrics = eval(metrics)

    for metric in metrics:
        metric_type, k = metric.split('@')
        k = int(k)
        if metric_type == 'ndcg':
            _metrics_dict[metric] = ndcg_at_k(_rec_list, k)
        elif metric_type == 'hit':
            _metrics_dict[metric] = hit_at_k(_rec_list, k)

    return _metrics_dict


# for each sequence, generate "num_beams" candidate item
def get_rec_list(_predictions, _scores, _targets, num_beams, _all_items):
    _rec_list = []
    batch_size = len(_targets)

    # for items that don't actually exist, set their scores to -inf
    for idx, indices in enumerate(_all_items):
        if indices not in _all_items:
            scores[idx] = float('-inf')

    for b in range(batch_size):
        batch_predictions = _predictions[b * num_beams:(b + 1) * num_beams]
        batch_scores = _scores[b * num_beams:(b + 1) * num_beams]
        target = targets[b]

        pairs = [(a, b) for a, b in zip(batch_predictions, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        one_rec = []
        for one_res in sorted_pairs:
            if one_res[0] == target:
                one_rec.append(1)
            else:
                one_rec.append(0)

        _rec_list.append(one_rec)

    return _rec_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ckpt & dataset
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--dataset', type=str, default='Games')
    parser.add_argument('--device', type=str, default='cuda:5' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--plm_dir', type=str, default='../LLM/')
    parser.add_argument('--plm_name', type=str, default='t5-base')
    parser.add_argument('--tokenizer_plm', type=str, default='sentence-t5-base')
    parser.add_argument('--token_type', type=str, default='pretrained_nc', choices=['sid', 'cid', 'pretrained', 'sid_nc', 'pretrained_nc'])

    # hyper-param
    parser.add_argument('--num_beams', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--K', type=int, default=256, help='codebook size')
    parser.add_argument('--D', type=int, default=3, help='codebook num')
    parser.add_argument('--item_sep', type=str, default=',', help='separation between two item token sequence')
    parser.add_argument('--max_len', type=int, default=20, help='max length of user interation sequence')
    parser.add_argument('--max_sent_len', type=int, default=512, help='max length of model input sequence')
    parser.add_argument('--add_user_prefix', type=bool, default=False,
                        help='whether add user prefix in the begin of sequence')
    parser.add_argument('--user_prefix', type=str, default='<u_{}>', )

    # metrics
    parser.add_argument('--metrics', type=str, default="['hit@5', 'hit@10', 'ndcg@5', 'ndcg@10']")

    args = parser.parse_args()

    set_seed()
    setup_logging()

    logging.info('hyper-param:')
    for param, value in vars(args).items():
        logging.info("  %s: %s", param, value)

    # load checkpoint
    ckpt_dir = os.path.join(args.ckpt_dir, args.token_type)
    if args.token_type == 'sid':
        dataset_name = args.dataset + '_' + args.tokenizer_plm
    else:
        dataset_name = args.dataset
    ckpt_dir = os.path.join(ckpt_dir, dataset_name)
    token_size = f'K={args.K}_D={args.D}'
    ckpt_dir = os.path.join(ckpt_dir, token_size)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16)

    model.to(args.device)
    model.eval()

    test_dataset = RecDataset(args, mode='test')

    all_items = test_dataset.get_all_items()

    candidate_trie = Trie(
        [
            [0] + tokenizer.encode(candidate) for candidate in all_items
        ]
    )
    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn(candidate_trie)

    collator = RecCollator(args, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collator, pin_memory=True,
                                 num_workers=4)

    metrics_dict = defaultdict(float)
    total = len(test_dataset)

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='[Testing]', disable=True):
            input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['labels']
            input_ids, attention_mask = input_ids.to(args.device), attention_mask.to(args.device)

            # for the meaning of each parameter, please refer to https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )

            output_ids = output['sequences']
            # the credibility of the generated indices, also can be regarded as the user's preference for the item in the recommendation system(higher is better).
            # 分数是概率经过log后的结果，所以都是负数。由于log是单调函数，即概率越大，score越大(越接近于0)
            scores = output['sequences_scores']

            predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            targets = tokenizer.batch_decode(targets, skip_special_tokens=True)

            scores = scores.detach().tolist()
            rec_list = get_rec_list(predictions, scores, targets, args.num_beams, all_items)
            batch_metrics_dict = get_metrics(args.metrics, rec_list)

            for metric, value in batch_metrics_dict.items():
                metrics_dict[metric] += value

    logging.info('evaluation result:')
    for metric, value in metrics_dict.items():
        logging.info("  %s: %s", metric, value / total)
