import argparse

import transformers.utils.import_utils
import logging

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoConfig, \
    EarlyStoppingCallback, T5Tokenizer, T5ForConditionalGeneration
from utils import RecDataset, ensure_dir, RecCollator, set_seed, setup_logging
import os

# needs to be set to false, otherwise deadlock may occur
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# setting constrained cuda device
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TrainerArguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--bf16', type=bool, default=transformers.utils.import_utils.is_torch_bf16_available())
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--optimizer', type=str, default='adamw_torch')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)

    # dataset & model
    parser.add_argument('--plm_dir', type=str, default='../LLM/')
    parser.add_argument('--plm_name', type=str, default='t5-base')
    parser.add_argument('--tokenizer_plm', type=str, default='sentence-t5-base')
    parser.add_argument('--dataset', type=str, default='Games')
    parser.add_argument('--token_type', type=str, default='pretrained_nc', choices=['sid', 'pretrained', 'cid', 'sid_nc'])

    # other hyper-param
    parser.add_argument('--K', type=int, default=256, help='codebook size')
    parser.add_argument('--D', type=int, default=3, help='codebook num')
    parser.add_argument('--add_user_prefix', type=bool, default=False,
                        help='whether add user prefix in the begin of sequence')
    parser.add_argument('--user_prefix', type=str, default='<u_{}>', help='user prefix template')
    parser.add_argument('--item_sep', type=str, default=',', help='separation between two item token sequence')
    parser.add_argument('--max_len', type=int, default=20, help='max length of user interation sequence')
    parser.add_argument('--max_sent_len', type=int, default=512, help='max length of model input sequence')

    args = parser.parse_args()
    set_seed()
    setup_logging()

    # get the serial number of each process in the host
    local_rank = int(os.environ.get('LOCAL_RANK') or 0)

    # setting output dir
    output_dir = os.path.join(args.output_dir, args.token_type)
    if args.token_type == 'sid':
        dataset_name = args.dataset + '_' + args.tokenizer_plm
    else:
        dataset_name = args.dataset
    output_dir = os.path.join(output_dir, dataset_name)
    token_size = f'K={args.K}_D={args.D}'
    output_dir = os.path.join(output_dir, token_size)
    ensure_dir(output_dir)

    # load data
    train_dataset, valid_dataset = RecDataset(args, mode='train'), RecDataset(args, mode='valid')

    # load config, tokenizer & model
    model_path = os.path.join(args.plm_dir, args.plm_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # treat each token in indices and user as a new token
    add_num = tokenizer.add_tokens(train_dataset.get_new_tokens())

    # change config, more specifically vocab_size in config
    config = AutoConfig.from_pretrained(model_path)
    config.vocab_size = len(tokenizer)

    # if process number is 0, save new config & tokenizer(avoid duplication)
    if local_rank == 0:
        tokenizer.save_pretrained(output_dir)
        config.save_pretrained(output_dir)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    # the newly added tokens will be initialized
    model.resize_token_embeddings(len(tokenizer))

    # print(model)

    collator = RecCollator(args, tokenizer=tokenizer)

    # logger
    logging.info("hyper-param:")
    for param, value in vars(args).items():
        logging.info("  %s: %s", param, value)

    # for training, doesn't use recommendation metric to evaluate, just use loss in valid_dataset
    # train
    training_args = Seq2SeqTrainingArguments(output_dir=output_dir,
                                             seed=2024,
                                             disable_tqdm=True,
                                             per_device_train_batch_size=args.batch_size,
                                             per_device_eval_batch_size=args.batch_size,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             num_train_epochs=args.epoch,
                                             optim=args.optimizer,
                                             learning_rate=args.lr,
                                             weight_decay=args.weight_decay,
                                             warmup_ratio=args.warmup_ratio,
                                             lr_scheduler_type=args.lr_scheduler_type,
                                             fp16=not args.bf16,
                                             bf16=args.bf16,
                                             dataloader_num_workers=4,
                                             save_strategy='epoch',
                                             evaluation_strategy='epoch',
                                             save_total_limit=2,
                                             load_best_model_at_end=True)
    # set false when training, see https://stackoverflow.com/questions/76633335/why-does-hugging-face-falcon-model-use-mode-config-use-cache-false-why-wouldn for more details
    model.config.use_cache = False

    trainer = Seq2SeqTrainer(model=model,
                             train_dataset=train_dataset,
                             eval_dataset=valid_dataset,
                             args=training_args,
                             tokenizer=tokenizer,
                             data_collator=collator,
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=10)])

    trainer.train()

    # save state
    trainer.save_state()
    # save best model
    trainer.save_model(output_dir=output_dir)
