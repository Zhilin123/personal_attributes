#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import metrics
from collections import defaultdict, Counter

#"../eval_tokens/head_reln_tail-not_within_sentence-new/discriminator_eval_tokens_epoch_4.499953170366208.csv"

filename = "../eval_tokens/head_reln_tail-within_sentence-new/discriminator_eval_tokens_epoch_6.249971972046788.csv"

df= pd.read_csv(filename)

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

def precision_recall_fscore(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1 = (2 * precision * recall) / (precision+recall +1e-12) 
    return precision, recall, f1


ground_reln_to_all_ground = defaultdict(list)
ground_reln_to_all_predicted = defaultdict(list)
ground_reln_to_ground_tail = defaultdict(list)
ground_reln_to_predicted_tail = defaultdict(list)
ground_reln_to_predicted_reln = defaultdict(list)
for i in range(len(ground_reln)):
    ground_reln_to_all_ground[ground_reln[i]].append(all_ground[i])
    ground_reln_to_all_predicted[ground_reln[i]].append(all_predicted[i])
    ground_reln_to_ground_tail[ground_reln[i]].append(ground_tail[i])
    ground_reln_to_predicted_tail[ground_reln[i]].append(predicted_tail[i])
    ground_reln_to_predicted_reln[ground_reln[i]].append(predicted_reln[i])
    
reln_to_data = defaultdict(dict)

for reln in ground_reln_to_all_ground:
    y_true = ground_reln_to_all_ground[reln]
    y_pred = ground_reln_to_all_predicted[reln]
    precision, recall, f1 = precision_recall_fscore(y_true, y_pred)
    
    reln_to_data[reln] = {
                "n":str(round(100*len(y_true)/len(all_ground),2)),
                "p":round(precision*100,1),
                "r":round(recall*100,1),
                "f1":round(f1*100,1),
                "predicted_tail":ground_reln_to_predicted_tail[reln],
                "ground_tail":ground_reln_to_ground_tail[reln],
                "predicted_reln":ground_reln_to_predicted_reln[reln]
            }
    
    #print(reln)
    #print(reln_to_data)
    #print(round(precision*100,1),'&', round(recall*100,1),'&',  round(f1*100,1) )

reln_by_f1 = sorted(list(reln_to_data.keys()), key=lambda reln:reln_to_data[reln]["f1"], reverse=True)
reln_by_n = sorted(list(reln_to_data.keys()), key=lambda reln:reln_to_data[reln]["n"], reverse=True)

def get_n_most_common(some_list, n=3, percent=False):
    most_common = [Counter(some_list).most_common(n)[-1]]
    #print(most_common)
    #print(round(100*i[1]/len(some_list),1))
    if not percent:
        return ', ' .join([i[0].replace('_','\_') for i in most_common])
    else:
        ## replace i[1] with round(100*i[1]/len(some_list),1) to get percentage
        #
        return ', ' .join(["{} ({})".format(i[0].replace('_','\_').replace('[', '{[').replace(']', ']}'), i[1]) for i in most_common])
for i in range(5):
    reln = reln_by_n[i]
    #print(reln_to_data[reln])
    data = reln_to_data[reln]
    useful_info = [
                '', '{'+reln.replace('_','\_')+'}', data['n'], data['p'], data['r'],data['f1'],
                get_n_most_common(data['predicted_reln'], percent=True, n=1),
                get_n_most_common(data['ground_tail'], percent=True, n=1),
                get_n_most_common(data['predicted_tail'], percent=True, n=1),
                   ]
    
    
    print(' & '.join(str(i) for i in useful_info), '\\\\')
    
    for j in range(2,4):
        useful_info = [
                '&'*5,
                get_n_most_common(data['predicted_reln'], percent=True, n=j),
                get_n_most_common(data['ground_tail'], percent=True, n=j),
                get_n_most_common(data['predicted_tail'], percent=True, n=j),
                   ]
    
    
        print(' & '.join(str(i) for i in useful_info), '\\\\')
    
    print("\\cline{2-9}")
    
    

