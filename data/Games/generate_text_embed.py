import argparse
import json
from collections import defaultdict

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import os
from torch.utils.data import DataLoader
import torch
from utils import clean_text, TextDataset



# concatenate title and description, different from LC-Rec
def get_item_text(_args):
    data_path = './Games.item.json'
    with open(data_path, 'r') as f:
        data = json.load(f)

    item_text_dict = defaultdict(str)
    features = eval(_args.features)
    for item_id, meta_data in data.items():
        text = ''
        for feature in features:
            if feature in meta_data:
                meta_value = clean_text(meta_data[feature])
                text += meta_value
        item_text_dict[int(item_id)] = text

    # sort by id
    sorted_dict = sorted(item_text_dict.items())
    _item_text_list = []
    for _, text in sorted_dict:
        _item_text_list.append(text)

    return _item_text_list


# not yet implemented
def generate_text_embed_by_bert():
    return None


# priority implemented
def generate_text_embed_by_sentencet5(_args, _model, _dataloader):
    _text_embeds = []

    # setting output type
    if _args.embed_type == 'mean':
        _model[1].pooling_mode_mean_token = True
        _model[1].pooling_mode_cls_token = False
    elif _args.embed_type == '<CLS>':
        _model[1].pooling_mode_cls_token = True
        _model[1].pooling_mode_mean_token = False

    with torch.no_grad():
        for text in _dataloader:
            outputs = _model.encode(text)
            _text_embeds.append(torch.tensor(outputs))

    _text_embeds = torch.concat(_text_embeds, dim=0).cpu()
    return _text_embeds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # device
    parser.add_argument('--device', type=str, default='cuda:4' if torch.cuda.is_available() else 'cpu')

    # hyper-param
    parser.add_argument('--embed_type', type=str, default='mean', choices=['mean', '<CLS>'])
    parser.add_argument('--features', type=str, default="['title', 'description']")
    parser.add_argument('--batch_size', type=int, default=1024)

    # lm
    parser.add_argument('--max_sent_len', type=int, default=512)
    parser.add_argument('--plm_dir', type=str, default='../../../LLM/')
    parser.add_argument('--plm_name', type=str, default='sentence-t5-base')

    args = parser.parse_args()

    item_text_list = get_item_text(args)
    dataset = TextDataset(item_text_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model_path = os.path.join(args.plm_dir, args.plm_name)

    if args.plm_name == 'bert-base-uncased':
        raise NotImplementedError('This function is not implemented yet.')
        # tokenizer, model = AutoTokenizer.from_pretrained(model_path), AutoModel.from_pretrained(model_path)
        # model = model.to(args.device)
        # text_embeds = generate_text_embed_by_bert(args, model, tokenizer, dataloader)
    elif args.plm_name == 'sentence-t5-base':
        model = SentenceTransformer(model_path)
        model = model.to(args.device)
        text_embeds = generate_text_embed_by_sentencet5(args, model, dataloader)
    else:
        raise NotImplementedError('this kind of text encoder is not implemented yet')

    text_embed_path = f'text_embed_{args.plm_name}.pt'
    torch.save(text_embeds, text_embed_path)
