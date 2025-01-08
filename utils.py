from torch.utils.data import Dataset
import os
import json
import random
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Dict
import html
import re
import logging
import sys


# 字典树
class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
            prefix_sequence: List[int],
            trie_dict: Dict,
            append_trie=None,
            bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


# Dataset
class TextDataset(Dataset):
    def __init__(self, x):
        super(TextDataset, self).__init__()
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)


class TokenizerDataset(Dataset):
    def __init__(self, x):
        super(TokenizerDataset, self).__init__()
        self.x = x

    def __getitem__(self, index):
        return self.x[index, :]

    def __len__(self):
        return self.x.shape[0]


class RecCollator(object):
    def __init__(self, args, tokenizer):
        super(RecCollator, self).__init__()
        self.tokenizer = tokenizer
        self.max_sent_len = args.max_sent_len

    # as a function
    def __call__(self, batch):
        input_ids = [one_data['input_ids'] for one_data in batch]
        labels = [one_data['labels'] for one_data in batch]

        inputs = self.tokenizer(text=input_ids,
                                text_target=labels,
                                return_tensors='pt',
                                padding='longest',
                                max_length=self.max_sent_len,
                                truncation=True,
                                return_attention_mask=True)

        return inputs


class RecDataset(Dataset):
    def __init__(self, args, mode='train', ):
        super(RecDataset, self).__init__()

        self.dataset = args.dataset
        self.tokenizer_plm = args.tokenizer_plm
        self.K = args.K
        self.D = args.D
        self.add_user_prefix = args.add_user_prefix
        if self.add_user_prefix:
            self.user_prefix = args.user_prefix
        self.item_sep = args.item_sep
        self.max_len = args.max_len
        self.token_type = args.token_type

        self.new_tokens = None
        self.all_items = None
        self.prefix_allowed_tokens = None

        self._load_data()
        self._id2token()

        if mode == 'train':
            self.data = self._get_train_data()
        elif mode == 'valid':
            self.data = self._get_valid_data()
        elif mode == 'test':
            self.data = self._get_test_data()

    # load train & indices mapping
    def _load_data(self):
        with open(f'./data/{self.dataset}/inter_data.json', 'r') as fp:
            self.inter_data = json.load(fp)

        if self.token_type == 'cid':
            with open(f'./tokenizer/cid_result/{self.dataset}/K={self.K}_D={self.D}_index.json',
                      'r') as fp:
                self.indices = json.load(fp)
        elif self.token_type == 'sid':
            with open(f'./tokenizer/sid_result/{self.dataset}_{self.tokenizer_plm}/K={self.K}_D={self.D}_index.json',
                      'r') as fp:
                self.indices = json.load(fp)
        elif self.token_type == 'pretrained':
            with open(f'./tokenizer/pretrained_result/{self.dataset}/K={self.K}_D={self.D}_index.json',
                      'r') as fp:
                self.indices = json.load(fp)
        elif self.token_type == 'sid_nc':
            with open(
                    f'./tokenizer/sid_result/{self.dataset}_{self.tokenizer_plm}/K={self.K}_D={self.D}_index_nc.json',
                    'r') as fp:
                self.indices = json.load(fp)
        elif self.token_type == 'pretrained_nc':
            with open(f'./tokenizer/pretrained_result/{self.dataset}/K={self.K}_D={self.D}_index_nc.json',
                      'r') as fp:
                self.indices = json.load(fp)

    # map id to token
    def _id2token(self):
        self.remapped_inter_data = dict()
        for user, seq in self.inter_data.items():
            self.remapped_inter_data[user] = [''.join(self.indices[str(item)]) for item in seq]

    # using indices get all tokens
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        # add item token
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)

        # add user token
        if self.add_user_prefix:
            for user in range(len(self.inter_data)):
                self.new_tokens.add(self.user_prefix.format(user))

        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    # for test, get all item indices for evaluating
    def get_all_items(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    # using slide window to get training data
    # Similar data augmentation methods have also been applied in RecBole
    def _get_train_data(self):
        # each piece of data is a dict
        data = []

        # leave-one-out
        for user, seq in self.remapped_inter_data.items():
            if self.add_user_prefix:
                user_prefix = self.user_prefix.format(user)
            train = seq[:-2]
            for i in range(1, len(train)):
                one_data = dict()
                history, target = train[:i], train[i]
                # truncation
                history = history[-self.max_len:]

                if self.add_user_prefix:
                    history = [user_prefix] + history

                # transform list to str using separation
                one_data['history'] = self.item_sep.join(history)
                one_data['target'] = target
                data.append(one_data)

        return data

    # the second to last one is used as validation set
    def _get_valid_data(self):
        data = []

        for user, seq in self.remapped_inter_data.items():
            if self.add_user_prefix:
                user_prefix = self.user_prefix.format(user)
            one_data = dict()
            history, target = seq[-self.max_len:-2], seq[-2]

            if self.add_user_prefix:
                history = [user_prefix] + history

            one_data['history'] = self.item_sep.join(history)
            one_data['target'] = target
            data.append(one_data)

        return data

    # the last one is used as test set
    def _get_test_data(self):
        data = []

        for user, seq in self.remapped_inter_data.items():
            if self.add_user_prefix:
                user_prefix = self.user_prefix.format(user)
            one_data = dict()
            history, target = seq[-self.max_len:-1], seq[-1]

            if self.add_user_prefix:
                history = [user_prefix] + history

            one_data['history'] = self.item_sep.join(history)
            one_data['target'] = target
            data.append(one_data)

        return data

    def __getitem__(self, index):

        data = self.data[index]

        # key must be 'input_ids' & 'labels' for huggingface to classify
        return {
            'input_ids': data['history'],
            'labels': data['target']
        }

    def __len__(self):
        return len(self.data)


# util functions
def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(raw_text):
    if isinstance(raw_text, list):
        new_raw_text = []
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r'</?\w+[^>]*>', '', raw)
            raw = re.sub(r'["\n\r]*', '', raw)
            new_raw_text.append(raw.strip())
        cleaned_text = ' '.join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r'</?\w+[^>]*>', '', cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def get_local_time():
    cur = datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def setup_logging(log_file=None, level=logging.INFO):
    """
    设置日志配置。

    Parameters:
    - log_file (str): 如果提供，将日志输出到该文件；否则输出到控制台。
    - level (int): 日志级别，默认是 logging.INFO。
    """
    handlers = [logging.StreamHandler(sys.stdout)]  # 默认输出到控制台

    # 如果指定了日志文件，将日志同时写入文件
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
