#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sklearn
from sklearn import metrics
import argparse
from tqdm import trange
import seaborn as sns
import matplotlib.pyplot as plt

#parser = argparse.ArgumentParser(description='get precision recall f1 from eval tokens file')
#parser.add_argument("--eval_tokens_filename")
#args = parser.parse_args()

#

dataset = "within_sentence-new"

def get_correct_indices(eval_tokens_filename, tail_filename=None, mode="all"):
    
    df= pd.read_csv(eval_tokens_filename)
    #print(len(df))
    ground_head = list(df["ground_head"])
    predicted_head = list(df["predicted_head"])
    ground_reln = list(df["ground_reln"])
    predicted_reln = list(df["predicted_reln"])
    ground_tail = list(df["ground_tail"])
    #ground_sentence = list(df['ground_sentence'])
    
    if tail_filename is not None:
        df = pd.read_csv(tail_filename)
    
    predicted_tail= list(df["predicted_tail"])
    

    ground_all = [ground_head[i] + ground_reln[i] + ground_tail[i] for i in range(len(ground_head))]
    predicted_all = [str(predicted_head[i]) + str(predicted_reln[i]) + str(predicted_tail[i]) for i in range(len(ground_head))]
    
    if mode == "all":
        y_pred = predicted_all
        y_true = ground_all
    
    elif mode == "head":
        y_pred = predicted_head
        y_true = ground_head
    
    elif mode == "reln":
        y_pred = predicted_reln
        y_true = ground_reln
    
    elif mode == "tail":
        y_pred = predicted_tail
        y_true = ground_tail
        
    #return y_pred, y_true
    if mode == "number":
        return len(ground_head)
    
    return [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]



all_correct = []
head_correct = []
reln_correct = []
tail_correct = []
tail_correct_only_tail =[]
all_combinations_correct = []

for j in trange(2,12):
    all_correct_indices = []
    tail_correct_indices = []
    reln_correct_indices = []
    head_correct_indices = []
    tail_only_correct_indices = []
    all_combinations_correct_indices = []
    
    all_filenames = ["../eval_tokens/head_reln_tail-{}/eval_tokens_epoch_first-{}.csv".format(dataset, i) for i in range(1,j)]
    #tail-within_sentence
    for filename in all_filenames:
        #y_pred, y_true = get_correct_indices(filename, tail_filename="eval_tokens/tail-within_sentence/eval_tokens_epoch_first-1.csv")
        all_correct_indices += get_correct_indices(filename, mode="all") #
        tail_correct_indices += get_correct_indices(filename, mode="tail")
        reln_correct_indices += get_correct_indices(filename, mode="reln")
        head_correct_indices += get_correct_indices(filename, mode="head")
        
        tail_only_correct_indices += get_correct_indices(filename.replace(
                                        "head_reln_tail",
                                        "tail"
                                    ), mode="tail")
    for i in range(1, j):
        for k in range(1,j):
            filename = "../eval_tokens/composite_eval_tokens_{}/composite_head_reln_{}_tail_{}.csv".format(dataset, i, k)
            all_combinations_correct_indices += get_correct_indices(filename, mode="all")
    
    
        
        
        
    all_correct.append(len(set(all_correct_indices)))
    head_correct.append(len(set(head_correct_indices)))
    reln_correct.append(len(set(reln_correct_indices)))
    tail_correct.append(len(set(tail_correct_indices)))
    tail_correct_only_tail.append(len(set(tail_only_correct_indices)))
    all_combinations_correct.append(len(set(all_combinations_correct_indices)))



total_n = get_correct_indices(filename, mode="number")


y = head_correct + reln_correct +  tail_correct + all_correct #tail_correct_only_tail + all_combinations_correct +

y = [i /total_n*100 for i in y]

x = [i for i in range(1,11)] * (len(y)//10)

hue =  ["head entity"] * 10 + ["relation"] * 10 + \
     ["tail entity"] * 10 + ["all"] * 10  #+ ["tail entity (gen)"] * 10 +["all correct (gen)"] * 10  # these are non-tail gen

title = "Upper bound on recall based on top n generation \n candidates considered (Extraction dataset)"
title = ""

palette ={
          "head entity": "black",
          "relation": "green",
          "tail entity": "blue",
          "all": "orange",
          "all correct (gen)":"purple",
          "tail entity (gen)": "red",
          }

                 
ax = sns.lineplot(x=x, y=y, hue=hue, palette=palette)

ax.set_title(title)
ax.set_xlabel("top n generation candidates considered")
ax.set_ylabel("Upper bound on recall (%)")
ax.set(ylim=(0, 100))
plt.rcParams['savefig.dpi'] = 300
plt.savefig('save_as_a_png.png')
#y_true = all_ground
#y_pred = all_predicted
#
#precision, recall, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted')
#accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
#print(round(precision*100,1),'&', round(recall*100,1),'&',  round(f1*100,1) )
