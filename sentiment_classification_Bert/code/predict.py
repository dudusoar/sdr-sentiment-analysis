#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
import pandas as pd
import argparse

from utils.function_utils import set_seed,get_device,get_label_map
from utils.model_utils import XFBert,XFRoberta

from transformers import AutoTokenizer

MODEL_DIR = {
    "bert-base-wwm": "bert-base-uncased",  # Use English BERT model for English text
}

def set_args():
    parser = argparse.ArgumentParser()
    # the general
    parser.add_argument('--seed',  default=1234, type=int, help="random seed for initialization")
    parser.add_argument("--epoches", default=30, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--max_seq_length", default=270, type=int,help="The maximum total input sequence length after tokenization.")

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
    parser.add_argument("--use_swa",default='False',type=str)

    # args generate params
    args = parser.parse_args()
    return args


def predict(args, model, input_text, device,labels_name=None):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        intent_logits = model(input_ids, attention_mask=attention_mask)
        intent_logits = F.softmax(intent_logits,dim=1)
        intent_pred = torch.argmax(intent_logits, dim=1).item()
        result = labels_name[intent_pred]
    return result


if __name__=="__main__":
    args = set_args()
    set_seed(args.seed)
    device = get_device()

    id2intent = {0: 'posi', 1: 'neg', 2: 'neutral'}
    intent2id = {'posi': 0, 'neg': 1, 'neutral': 2}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR[args.MODEL_NAME])

    # Load model
    if "robert" in args.MODEL_NAME:
        model = XFRoberta(MODEL_DIR[args.MODEL_NAME], intent_dim=len(intent2id))
    else:
        model = XFBert(MODEL_DIR[args.MODEL_NAME], intent_dim=len(intent2id), enable_mdrop=args.enable_mdrop)
    # Model parameters
    args.save_dir_curr = os.path.join("../user_data/checkpoint", 'parameter_{}.pkl'.format(args.MODEL_NAME))
    model.load_state_dict(torch.load(args.save_dir_curr))
    # Model parameters
    model.to(device)

    input_text = ""
    result = predict(args, model, input_text, device, labels_name=id2intent)
    print('Input text:',input_text,'\n','Prediction:',result)



