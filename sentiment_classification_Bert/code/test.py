#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import torch
import torch.nn.functional as F
import pandas as pd
import argparse

from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

from torch.utils.data import DataLoader, SequentialSampler
from utils.function_utils import set_seed,get_label_map,get_device
from utils.model_utils import XFBert,XFRoberta
# from utils.model_utils_orign import XFBert,XFRoberta
from preprocess.processor import Tags_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

import numpy as np

MODEL_DIR = {
    "bert-base-wwm": "bert-base-uncased",  # Use English BERT model for English text
}

def set_args():
    parser = argparse.ArgumentParser()
    # the general
    parser.add_argument('--seed',  default=1234, type=int, help="random seed for initialization")
    parser.add_argument("--epoches", default=30, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--valid_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--test_batch_size", default=64, type=int, help="Batch size for test.")

    # the data
    parser.add_argument("--train_dir",  default=r"../data/comments_new/train.csv",type=str)
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

    # The tools
    parser.add_argument("--ema_decay",default=0.999,type=float)
    parser.add_argument("--use_ema",default=False,type=bool)
    # args generate params
    args = parser.parse_args()
    return args


def evaluation(args, model, data_loader, device, ema=None,labels_name=None):
    model.eval()
    if args.use_ema:
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
    labels = None
    prob_preds = None
    for batch in tqdm(data_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_labels = batch['intent_labels'].to(device)
            intent_logits = model(input_ids, attention_mask=attention_mask)
            intent_logits = F.softmax(intent_logits,dim=1)
            # Calculate slot prediction labels and true labels
            prob_preds = intent_logits.detach().cpu().numpy()
            labels = intent_labels.detach().cpu().numpy()

    intent_acc = accuracy_score(y_true=labels, y_pred=np.argmax(prob_preds,axis=1))
    preds = np.argmax(prob_preds, axis=1)
    precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    print("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(intent_acc, precision, recall, f1))
    return intent_acc


if __name__=="__main__":
    args = set_args()
    train = pd.read_csv(args.train_dir, sep=',')
    test = pd.read_csv(args.test_dir,sep=',')
    set_seed(args.seed)
    device = get_device()

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

    # Pass generator to load_dataset
    intent_valid_ds = Tags_dataset(data=test,tokenizer=tokenizer,max_seq_len=args.max_seq_length,intent2id=intent2id)
    val_sampler = SequentialSampler(intent_valid_ds)
    val_dataloader = DataLoader(intent_valid_ds,batch_size=args.valid_batch_size,sampler=val_sampler,drop_last=False, pin_memory=True)

    # Load model
    model = XFBert(MODEL_DIR[args.MODEL_NAME], intent_dim=len(intent2id), enable_mdrop=args.enable_mdrop)
    # Model parameters
    args.save_dir_curr = os.path.join("../user_data/checkpoint", 'parameter_{}.pkl'.format(args.MODEL_NAME))
    model.load_state_dict(torch.load(args.save_dir_curr))
    model.to(device)

    best_score = evaluation(args, model, val_dataloader, device)
    print('Test score:',best_score)



