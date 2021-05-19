#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from ast import literal_eval
from collections import Counter
from sklearn import metrics
import seaborn as sns
from matplotlib.colors import LogNorm
import numpy as np
import argparse
import matplotlib.pyplot as plt
import spacy

parser = argparse.ArgumentParser(description='ling analysis')
parser.add_argument("--csv_filename", help="csv file containing eval tokens with analysis")
parser.add_argument("--interested_field", default='reln', choices=['reln','big_pos_tags','dependency_labels', "reln_descriptive"])
args = parser.parse_args()


csv_filename = args.csv_filename
interested_field =  args.interested_field

df = pd.read_csv(csv_filename)

def get_correct_ids_reln(df):
    predicted_tokens = list(df['predicted_reln'])
    ground_truth_tokens = list(df['ground_reln'])
    return set([i for i in range(len(predicted_tokens)) if predicted_tokens[i] == ground_truth_tokens[i]])


def get_correct_ids_tail(df):
    predicted_tokens = list(df['predicted_tail'])
    ground_truth_tokens = list(df['ground_tail'])
    return set([i for i in range(len(predicted_tokens)) if predicted_tokens[i] == ground_truth_tokens[i]])

def get_correct_ids(df):
    if interested_field == "reln":
        return get_correct_ids_reln(df)
    return get_correct_ids_tail(df)

def preprocess_pos_tags(predicted_or_ground="ground"):
    field = list(df['{}_tail_{}'.format(predicted_or_ground, interested_field)])
    field = [literal_eval(i) for i in field]
    field = [field[i] for i in range(len(field)) if i not in correct_ids]
    field = ["-".join(i) for i in field]
    return field

def get_tail_span(all_tokens_i, correct_tokens_i):
    tail_pos = None
    for i in range(len(all_tokens_i)-len(correct_tokens_i)):
        if all_tokens_i[i:i+len(correct_tokens_i)] == correct_tokens_i and tail_pos is None:
            tail_pos = (i, i+len(correct_tokens_i))
    return tail_pos

def preprocess_dependency_labels():
    all_tokens = [literal_eval(i) for i in list(df['ground_sentence_tokens'])]
    correct_tokens = [literal_eval(i) for i in list(df['ground_tail_tokens'])]
    all_dependency_labels = [literal_eval(i) for i in list(df['ground_sentence_dependency_labels'])]
    tail_spans = [get_tail_span(all_tokens[i], correct_tokens[i]) for i in range(len(all_tokens))]


    tail_dependency_labels = [all_dependency_labels[i][tail_spans[i][0]:tail_spans[i][1]] for i in range(len(tail_spans))\
                                if tail_spans[i] is not None ]

    tail_dependency_labels  = ["\n".join(i) for i in tail_dependency_labels]

    return tail_dependency_labels


def preprocess_reln(predicted_or_ground="ground"):
    field = list(df['{}_reln'.format(predicted_or_ground)])
    return field

def preprocess(predicted_or_ground="ground"):
    if interested_field == "reln":
        return preprocess_reln(predicted_or_ground=predicted_or_ground)
    elif interested_field == 'big_pos_tags':
        return preprocess_pos_tags(predicted_or_ground=predicted_or_ground)
    elif interested_field == 'dependency_labels':
        return preprocess_dependency_labels()
    elif interested_field == "reln_descriptive":
        return preprocess_reln(predicted_or_ground=predicted_or_ground)
    else:
        return ValueError

def keep_values_where_ground_and_predicted_are_in_most_common(predicted_field,
                                                              ground_truth_field,
                                                              most_common_twenty_field_names):
    useful_index = [i for i in range(len(predicted_field)) if \
                    predicted_field[i] in most_common_twenty_field_names and \
                    ground_truth_field[i] in  most_common_twenty_field_names]

    predicted_field = [predicted_field[i] for i in useful_index]
    ground_truth_field = [ground_truth_field[i] for i in useful_index]
    return predicted_field, ground_truth_field

def get_proportion_of_correct_field(data):
    return round(np.sum(np.diagonal(data)) / np.sum(data)*100,1)


def draw_confusion_matrix(data, labels):

    ax = sns.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'},fmt='g')

    # to log heatmap for dependency
    # use vmin=1, vmax=np.max(data), norm=LogNorm()
    # cbar_kws["ticks"] = [1,10,100,1000]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xticklabels(labels, rotation=90)

    ax.set_yticklabels(labels, rotation=0)

    if interested_field == 'big_pos_tags':
        print_field_name = "POS tags"
    elif interested_field == 'dependency_labels':
        print_field_name = "Dependency labels"
    elif interested_field == 'reln':
        print_field_name = 'Relation'

    ax.set(ylabel="True {}".format(print_field_name),
           xlabel="Predicted {}".format(print_field_name))

    plt.tight_layout()
    plt.show()

def draw_barplot(most_common_field_names, ground_truth_field,xlabel=""):
    x = [i[0].replace("-","\n-") for i in most_common_field_names]
    y = [i[1] for i in most_common_field_names]
    y = [100*i/len(ground_truth_field) for i in y]
    ax = sns.barplot(x=x, y=y)
    ax.set_xticklabels(x, rotation=30) #90
    ax.set(xlabel=xlabel, ylabel="Proportion (%)")
    plt.tight_layout()
    plt.show()

correct_ids = get_correct_ids(df)
ground_truth_field = preprocess(predicted_or_ground="ground")

if interested_field in ['reln']:

    predicted_field = preprocess(predicted_or_ground="predicted")
    most_common_field_names = Counter(predicted_field+ground_truth_field).most_common(10)
    most_common_field_names = [i[0] for i in most_common_field_names]

    predicted_field, ground_truth_field = keep_values_where_ground_and_predicted_are_in_most_common(predicted_field,
                                                                                                      ground_truth_field,
                                                                                                      most_common_field_names)
    data = metrics.confusion_matrix(ground_truth_field , predicted_field , labels = most_common_field_names)

    draw_confusion_matrix(data, most_common_field_names)

    print("Total Errors: {}".format(np.sum(data)))
    print("Correct {}: {} %".format(interested_field, get_proportion_of_correct_field(data)))

elif interested_field in ['dependency_labels', 'reln_descriptive','big_pos_tags']:
    top_k = 10
    most_common_field_names = Counter(ground_truth_field).most_common(top_k)
    if interested_field == 'dependency_labels':
        xlabel= "Dependency labels of tail_entity"  
    elif interested_field == 'big_pos_tags':
        xlabel= "POS Tags of tail_entity"  
    else: 
        xlabel= "Most common relations" 
    draw_barplot(most_common_field_names, ground_truth_field, xlabel=xlabel)
