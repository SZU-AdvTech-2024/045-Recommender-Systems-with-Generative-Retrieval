import argparse

import logging
import os.path

from model.rqvae import RQVAE
import torch
from utils import TokenizerDataset, set_seed
from torch.utils.data import DataLoader
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data/device
    parser.add_argument('--data_path', type=str, default="../data/")
    parser.add_argument('--device', type=str, default='cuda:6' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    # parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument('--ckpt_dir', type=str, default='saved_model/')
    parser.add_argument('--dataset', type=str, default='Games')
    parser.add_argument('--tokenizer_plm', type=str, default='sentence-t5-base')
    parser.add_argument('--token_type', type=str, default='pretrained', choices=['sid', 'pretrained'])

    # hyper-param
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--eval_step', type=int, default=50, help='calculate collision rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='l2 regularization')
    parser.add_argument('--dropout_prob', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--bn', type=bool, default=False, help='use batch normalization')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'l1'], help='loss type')
    # better for training LLM
    parser.add_argument('--lr_scheduler_type', type=str, default='constant')
    parser.add_argument('--warmup_epochs', type=int, default=50)

    parser.add_argument('--kmeans_init', type=bool, default=True,
                        help='whether to initialize kmeans cluster in first batch')
    parser.add_argument('--kmeans_iters', type=int, default=100, help='max kmeans iterations')

    # param for RQ-VAE
    # sid
    parser.add_argument('--layers', nargs='+', type=int, default=[2048, 1024, 512, 256, 128, 64],
                        help='each hidden size in encoder')
    parser.add_argument('--num_emb_list', nargs='+', type=int, default=[256, 256, 256],
                        help='codebook num and codebook size')
    parser.add_argument('--e_dim', type=int, default=32, help='dimension of code')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantization loss weight')
    parser.add_argument('--beta', type=float, default=0.25, help='beta for commitment loss')

    args = parser.parse_args()

    # for reproducibility
    set_seed()

    logging.basicConfig(level=logging.DEBUG)

    # read data
    data_path = os.path.join(args.data_path, args.dataset)
    # different token type
    if args.token_type == 'sid':
        file_name = f'text_embed_{args.tokenizer_plm}.pt'
    elif args.token_type == 'pretrained':
        file_name = f'pretrained_sasrec.pt'

    file_path = os.path.join(data_path, file_name)
    semantic_embed = torch.load(file_path)
    data = TokenizerDataset(semantic_embed)

    dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # build
    model = RQVAE(in_dim=semantic_embed.shape[-1],
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters)
    model = model.to(args.device)

    # train
    trainer = Trainer(args, model, len(dataloader))
    best_loss, best_collision_rate = trainer.fit(dataloader)

    print(f"K={args.num_emb_list[0]}, D={len(args.num_emb_list)}")
    print(f'best loss: {best_loss}, best collision rate: {best_collision_rate}')
