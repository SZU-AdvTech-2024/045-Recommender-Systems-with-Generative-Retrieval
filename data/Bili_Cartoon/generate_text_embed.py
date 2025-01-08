import argparse

from torch.utils.data import DataLoader

from utils import TextDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import torch


def get_item_text():
    data_path = './Bili_Cartoon_item.csv'
    data = pd.read_csv(data_path, names=['item_id', 'cn_title', 'en_title'])
    data.columns = ['item_id', 'cn_title', 'en_title']
    # raw data is shuffled
    data.sort_values(by=['item_id'], inplace=True)
    _item_text_list = data.iloc[:, 2].values.tolist()

    return _item_text_list


def generate_text_embed_by_bert(_args, _model, _tokenizer, _dataloader):
    _text_embeds = []

    # no compute grad
    with torch.no_grad():
        for text in tqdm(_dataloader):
            encoded_text = _tokenizer(text, max_length=_args.max_sent_len, truncation=True, return_tensors='pt',
                                      padding="longest").to(_args.device)
            # hidden_state in last layer
            outputs = _model(input_ids=encoded_text.input_ids, attention_mask=encoded_text.attention_mask)

            if _args.embed_type == 'mean':
                # compute avg but not include padding
                masked_output = outputs.last_hidden_state * encoded_text.attention_mask.unsqueeze(-1)
                mean_output = masked_output.sum(dim=1) / encoded_text.attention_mask.sum(dim=-1, keepdim=True)
                _text_embeds.append(mean_output)
            elif _args.embed_type == '<CLS>':
                # take <CLS> output as embedding
                cls_output = outputs.last_hidden_state[:, 0, :]
                _text_embeds.append(cls_output)

    # concat list of tensor
    _text_embeds = torch.concat(_text_embeds, dim=0).cpu()

    return text_embeds


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
        for text in tqdm(_dataloader):
            outputs = _model.encode(text)
            _text_embeds.append(torch.tensor(outputs))

    _text_embeds = torch.concat(_text_embeds, dim=0).cpu()
    return _text_embeds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # device
    parser.add_argument('--device', type=str, default='cuda:4' if torch.cuda.is_available() else 'cpu')

    # hyper_param
    parser.add_argument('--embed_type', type=str, default='mean', choices=['mean', '<CLS>'])
    parser.add_argument('--batch_size', type=int, default=1024)

    # lm
    parser.add_argument('--max_sent_len', type=int, default=512)
    parser.add_argument('--plm_dir', type=str, default='../../../LLM/')
    parser.add_argument('--plm_name', type=str, default='sentence-t5-base')

    args = parser.parse_args()

    item_text_list = get_item_text()
    dataset = TextDataset(item_text_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model_path = os.path.join(args.plm_dir, args.plm_name)

    if args.plm_name == 'bert-base-uncased':
        tokenizer, model = AutoTokenizer.from_pretrained(model_path), AutoModel.from_pretrained(model_path)
        model = model.to(args.device)
        text_embeds = generate_text_embed_by_bert(args, model, tokenizer, dataloader)
    elif args.plm_name == 'sentence-t5-base':
        model = SentenceTransformer(model_path)
        model = model.to(args.device)
        text_embeds = generate_text_embed_by_sentencet5(args, model, dataloader)
    else:
        raise NotImplementedError('this kind of text encoder is not implemented yet')

    text_embed_path = f'text_embed_{args.plm_name}.pt'
    torch.save(text_embeds, text_embed_path)
