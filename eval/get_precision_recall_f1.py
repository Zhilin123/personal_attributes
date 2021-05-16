#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sklearn
from sklearn import metrics
import argparse

parser = argparse.ArgumentParser(description='get precision recall f1 from eval tokens file')
parser.add_argument("--eval_tokens_filename")
parser.add_argument("--mode", default="all")
args = parser.parse_args()

eval_tokens_filename = args.eval_tokens_filename
mode = args.mode

df= pd.read_csv(eval_tokens_filename)

def process_col(col_name):
    return [str(i).strip() for i in list(df[col_name])]

ground_head = process_col("ground_head")
predicted_head = process_col("predicted_head")
ground_reln = process_col("ground_reln")
predicted_reln = process_col("predicted_reln")
ground_tail = process_col("ground_tail")
predicted_tail= process_col("predicted_tail")
ground_sentence = process_col("ground_sentence")


all_ground = [ground_head[i] + ground_reln[i] + ground_tail[i] for i in range(len(ground_head))]
all_predicted = [predicted_head[i] + predicted_reln[i] + predicted_tail[i] for i in range(len(ground_head))]

if mode == "all":
    y_true = all_ground
    y_pred = all_predicted
elif mode == "tail":
    y_true = ground_tail
    y_pred = predicted_tail
elif mode == "reln":
    y_true = ground_reln
    y_pred = predicted_reln
elif mode == "head":
    y_true = ground_head
    y_pred = predicted_head


def precision_recall_fscore(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, average='weighted',zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average='weighted',zero_division=0)
    f1 = (2 * precision * recall) / (precision+recall +1e-12)
    return precision, recall, f1

precision, recall, f1 = precision_recall_fscore(y_true, y_pred)
accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
print(round(precision*100,1),'&', round(recall*100,1),'&',  round(f1*100,1) )
