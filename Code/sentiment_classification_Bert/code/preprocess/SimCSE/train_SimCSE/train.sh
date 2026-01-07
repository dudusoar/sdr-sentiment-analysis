# !/bin/bash

python train_unsup.py \
--train_file ./data/Chinese_train.txt \
--max_length 128 \
--pretrained mc-bert \
--learning_rate 5e-6 \
--save_final True \
--tau 0.05 \