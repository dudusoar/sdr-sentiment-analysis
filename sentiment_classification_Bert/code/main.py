#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2' # Specify which GPU to use, or multiple GPUs, default is 0

import argparse
from argparse import ArgumentTypeError
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.function_utils import set_seed,get_time_dif,get_label_map,get_device
from utils.model_utils import XFBert,XFRoberta
# from utils.model_utils_orign import XFBert,XFRoberta
from preprocess.processor import Tags_dataset
from utils.trainer import get_optimizer_and_schedule,do_train,evaluation
from transformers import AutoTokenizer

# Import logging module
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
import warnings
warnings.filterwarnings('ignore')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

MODEL_DIR = {
    # "bert-base-wwm": "/home/ywc/pretrained_model/bert-base-uncased/",
    # "bert-base-wwm": "/home/ywc/intent_classifierr_4/SQL_CLS/code/preprocess/SimCSE/train_SimCSE/ckpts/bert_base_uncased_train-final/",
    "bert-base-wwm": "bert-base-uncased",  # Use English BERT model for English text
}

def set_args():
    parser = argparse.ArgumentParser()
    # the general
    parser.add_argument('--seed',  default=1234, type=int, help="random seed for initialization")
    parser.add_argument("--epoches", default=30, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size for training.")
    parser.add_argument("--valid_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--test_batch_size", default=64, type=int, help="Batch size for test.")

    # the data
    parser.add_argument("--train_dir",  default=r"../data/comments_new/train.csv",type=str)
    parser.add_argument("--val_dir", default=r"../data/comments_new/val.csv", type=str)
    parser.add_argument("--test_dir",default=r"../data/comments_new/test.csv", type=str)
    parser.add_argument("--max_seq_length", default=120, type=int,help="The maximum total input sequence length after tokenization.")

    # the model
    parser.add_argument("--MODEL_NAME", default="bert-base-wwm", type=str,help="The model name")
    parser.add_argument("--learning_rate", default=5e-5,type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="The initial learning rate for Weight decay.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--early_stopping",default=3,type=int)
    parser.add_argument("--scheduler_name",default="linear",type=str)
    parser.add_argument("--noise_lambda",default=0.0,type=float)
    parser.add_argument("--grad_accumulate_nums",default=1,type=int)
    parser.add_argument("--enable_mdrop",default=False,type=bool)

    # The path
    parser.add_argument("--save_dir_curr",default='../user_data/checkpoint',type=str)
    parser.add_argument("--results_excel_dir",default='../user_data/results',type=str,help="The path where training data is stored")

    # The state
    parser.add_argument("--is_train",default='yes',type=str)
    parser.add_argument("--is_predict", default='yes', type=str)

    # The tools
    parser.add_argument("--ema_decay",default=0.999,type=float)
    parser.add_argument("--use_ema",default=False,type=bool)
    parser.add_argument("--use_swa",default='False',type=str)

    # args generate params
    args = parser.parse_args()
    return args

def run(args):
    set_seed(args.seed)
    device = get_device()

    # View data
    train = pd.read_csv(args.train_dir,sep=',')
    valid = pd.read_csv(args.val_dir,sep=',')
    test = pd.read_csv(args.test_dir,sep=',')

    # Count labels
    INTENT_LIST = list(train['label'].unique())
    print("Test set has %d samples" % (test.shape[0]))
    print(f"Sentiment labels: {len(INTENT_LIST)} categories\nThey are: {INTENT_LIST}")

    # Label mapping dictionary
    id2intent = {0: 'posi', 1: 'neg', 2: 'neutral'}
    intent2id = {'posi': 0, 'neg': 1, 'neutral': 2}
    print(id2intent, intent2id)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR[args.MODEL_NAME])
    # Create test dataset
    test_valid_ds = Tags_dataset(data=test,
                                 tokenizer=tokenizer,
                                 max_seq_len=args.max_seq_length,
                                 prediction=True)
    test_sampler = SequentialSampler(test_valid_ds)
    test_dataloader = DataLoader(test_valid_ds,
                                 batch_size=args.test_batch_size,
                                 sampler=test_sampler,
                                 drop_last=False,
                                 pin_memory=True)

    print("Training set has %d samples" % (train.shape[0]))
    print("Validation set has %d samples" % (valid.shape[0]))
    print(train["label"].unique())
    print("***")
    # Pass generator to load_dataset
    intent_train_ds = Tags_dataset(data=train,tokenizer=tokenizer,max_seq_len=args.max_seq_length,intent2id=intent2id)
    intent_valid_ds = Tags_dataset(data=valid,tokenizer=tokenizer,max_seq_len=args.max_seq_length,intent2id=intent2id)
    train_sampler = RandomSampler(intent_train_ds, replacement=False)
    val_sampler = SequentialSampler(intent_valid_ds)
    train_loader = DataLoader(intent_train_ds,batch_size=args.train_batch_size,sampler=train_sampler,drop_last=False,pin_memory=True)
    val_dataloader = DataLoader(intent_valid_ds,batch_size=args.valid_batch_size,sampler=val_sampler,drop_last=False,pin_memory=True)
    # Load model
    if "robert" in args.MODEL_NAME:
        model = XFRoberta(MODEL_DIR[args.MODEL_NAME], intent_dim=len(intent2id))
    else:
        model = XFBert(MODEL_DIR[args.MODEL_NAME], intent_dim=len(intent2id), enable_mdrop=args.enable_mdrop)
    # Model parameters
    model.to(device)
    # optimizer, scheduler
    optimizer,scheduler = get_optimizer_and_schedule(args,model,trainloader_shape=len(train))
    torch.cuda.empty_cache()
    args.save_dir_curr = os.path.join("../user_data/checkpoint", 'parameter_{}.pkl'.format(args.MODEL_NAME))
    os.makedirs(os.path.dirname(args.save_dir_curr), exist_ok=True)
    # Initialize variables
    best_score = 0.0
    test_score = 0.0
    # Training
    if args.is_train == "yes":
        best_score = do_train(args, model, train_loader, val_dataloader, device, intent2id, optimizer, scheduler)
    # Load best model
    if args.is_predict == "yes":
        model.load_state_dict(torch.load(args.save_dir_curr))
        model.to(device)
        test_score = evaluation(args, model, test_dataloader, device)

    return best_score,test_score

if __name__=="__main__":
    start_time = time.time()
    logging.info('----------------Start timing----------------')
    logging.info('--------------------------------------------')
    args = set_args()

    best_score,test_score = run(args)
    print('best_score:{},test_score:{}'.format(best_score,test_score))

    time_dif = get_time_dif(start_time)
    logging.info("----------Container running time: {}-----------".format(time_dif))
