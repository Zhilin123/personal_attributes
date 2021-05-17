#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from collections import defaultdict, OrderedDict
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
import itertools
from tqdm import trange
import json
import argparse


parser = argparse.ArgumentParser(description='commonsense+semantic')
parser.add_argument("--csv_filename", help="csv file containing eval tokens with analysis")
parser.add_argument("--mode", default='dataset_analysis', choices=["dataset_analysis", "prediction_analysis"])
args = parser.parse_args()

porter_stemmer = PorterStemmer()

analysis_mode = args.mode

#Dataset analysis
if analysis_mode == "dataset_analysis":
    filename = "data/eval_sentences_tail_not_in_sentence.csv"
    filename_prefix = ["tail", "sentence"]
    conceptnet_connected_filenames = ["data/conceptnet_words/eval_{}_words_to_connected_words_all.json".format(i) for i in filename_prefix]
    conceptnet_related_filenames = ["data/conceptnet_words/eval_{}_words_to_related_words_all.json".format(i) for i in filename_prefix]

#Model prediction analysis
elif analysis_mode == "prediction_analysis":
    #filename = "../../nayak_not_within_sentence_eval.csv"
    filename = "data/discriminator_eval_tokens.csv"
    conceptnet_connected_filenames = ["data/conceptnet_words/eval_sentence_words_to_connected_words_all.json"]
    conceptnet_related_filenames = ["data/conceptnet_words/eval_sentence_words_to_related_words_all.json"]


tail_label = "ground_tail" if analysis_mode == "dataset_analysis" else "predicted_tail"

df = pd.read_csv(filename)
print("len: {}".format(len(df)))
ground_sentence = list(df["ground_sentence"])
tail = list(df[tail_label])

def load_conceptnet(conceptnet_filenames):
    # to account for when word is not in data
    word_to_conceptnet_associated_words = defaultdict(list)

    for conceptnet_filename in conceptnet_filenames:
        with open(conceptnet_filename, "r") as read_file:
           data = json.load(read_file)

        for i in data:
            word_to_conceptnet_associated_words[i] = data[i]

    return word_to_conceptnet_associated_words

def combine(defaultdict0, defaultdict1):
    new_defaultdict = defaultdict(list)
    all_keys = list(set(list(defaultdict0.keys()) + list(defaultdict1.keys())))
    for i in all_keys:
        new_defaultdict[i] = list(set(defaultdict0[i] + defaultdict1[i]))
    return new_defaultdict

word_to_conceptnet_connected_words = load_conceptnet(conceptnet_connected_filenames)
word_to_conceptnet_related_words = load_conceptnet(conceptnet_related_filenames)
word_to_conceptnet_related_or_connected_words = combine(word_to_conceptnet_connected_words,
                                                        word_to_conceptnet_related_words)

def is_contiguous_fragment(sentence_split, tail_split):
    for j in range(len(sentence_split)-len(tail_split)):
        if sentence_split[j:j+len(tail_split)] == list(tail_split):
            return 1
    return 0

def is_reorder(sentence_split, tail_split):
    for j in range(len(tail_split)):
        if tail_split[j] not in sentence_split:
            return 0
    return 1

def is_same_stem(sentence_split, tail_split):
    new_sentence_split = [porter_stemmer.stem(w) for w in sentence_split]
    new_tail_split = [porter_stemmer.stem(w) for w in tail_split]
    return is_contiguous_fragment(new_sentence_split, new_tail_split)

def is_same_stem_plus_reorder(sentence_split, tail_split):
    new_sentence_split = [porter_stemmer.stem(w) for w in sentence_split]
    new_tail_split = [porter_stemmer.stem(w) for w in tail_split]
    return is_reorder(new_sentence_split, new_tail_split)

def get_synonyms(tail_split):
    synonyms = []
    for i in range(len(tail_split)):
        synonyms.append([tail_split[i]])
        for syn in wn.synsets(tail_split[i]):
            for l in syn.lemmas():
                synonyms[i].append(l.name())

    synonyms = [list(set(i)) for i in synonyms]
    return synonyms

def get_hypernyms(tail_split):

    hypernyms = []
    for i in range(len(tail_split)):
        hypernyms.append([tail_split[i]])
        for syn in wn.synsets(tail_split[i]):
            for hyper in syn.hypernyms():
                for l in hyper.lemmas():
                    hypernyms[i].append(l.name())

    hypernyms = [list(set(i)) for i in hypernyms]
    return hypernyms

def get_hyponyms(tail_split):

    hyponyms = []
    for i in range(len(tail_split)):
        hyponyms.append([tail_split[i]])
        for syn in wn.synsets(tail_split[i]):
            for hyper in syn.hyponyms():
                for l in hyper.lemmas():
                    hyponyms[i].append(l.name())

    hyponyms = [list(set(i)) for i in hyponyms]

    return hyponyms

def combine_list_of_uniques(list_of_list_of_uniques):
    output_list = []

    for i in range(len(list_of_list_of_uniques[0])):
        one_list = []
        for j in range(len(list_of_list_of_uniques)):
            one_list += list_of_list_of_uniques[j][i]
        output_list.append(list(set(one_list)))

    return output_list

def get_wordnet(tail_split, mode="synonyms"):
    if mode == "synonyms":
        return get_synonyms(tail_split)
    elif mode == "hypernyms":
        return get_hypernyms(tail_split)
    elif mode == "hyponyms":
        return get_hyponyms(tail_split)
    elif mode == "all":
        return combine_list_of_uniques([
                    get_synonyms(tail_split),
                    get_hypernyms(tail_split),
                    get_hyponyms(tail_split),
                ])

def get_conceptnet_related_or_connected_words(tail_split):
    return [word_to_conceptnet_related_or_connected_words[w]+[w] for w in tail_split]

def get_conceptnet_related_words(tail_split):
    return [word_to_conceptnet_related_words[w]+[w] for w in tail_split]

def get_conceptnet_connected_words(tail_split):
    return [word_to_conceptnet_connected_words[w]+[w] for w in tail_split]


def get_conceptnet(tail_split, mode="related"):
    if mode == "related":
        return get_conceptnet_related_words(tail_split)
    elif mode == "related_or_connected":
        return get_conceptnet_related_or_connected_words(tail_split)
    elif mode == "connected":
        return get_conceptnet_connected_words(tail_split)

def get_field(tail_split, field="wordnet_synonyms"):
    if field.startswith("wordnet_"):
        return get_wordnet(tail_split, mode=field.replace("wordnet_",""))
    elif field.startswith("conceptnet_"):
        return get_conceptnet(tail_split, mode=field.replace("conceptnet_",""))

def get_field_in_sentence_split(sentence_split, tail_split, field="wordnet_synonyms"):
    field = get_field(tail_split, field=field)
    field_in_sentence_split = [[j for j in i if j in sentence_split] for i in field]
    return field_in_sentence_split

def get_combinations_of_nyms_in_sentence(sentence_split, tail_split, field="wordnet_synonyms"):
    field_in_sentence_split = get_field_in_sentence_split(sentence_split, tail_split, field=field)
    return [p for p in itertools.product(*field_in_sentence_split)]

def is_tail_in_field_plus_reorder(sentence_split, tail_split, field="conceptnet_connected"):
    sentence_field = get_field(sentence_split, field=field)
    all_words_in_sentence_field = set()
    for i in sentence_field:
        for j in i:
            all_words_in_sentence_field.add(j)
    for word in tail_split:
        if word not in all_words_in_sentence_field:
            return 0
    return 1

def is_tail_in_conceptnet_connected_plus_reorder(sentence_split, tail_split):
    return is_tail_in_field_plus_reorder(sentence_split, tail_split, field="conceptnet_connected")
def is_tail_in_conceptnet_related_plus_reorder(sentence_split, tail_split):
    return is_tail_in_field_plus_reorder(sentence_split, tail_split, field="conceptnet_related")
def is_tail_in_conceptnet_related_or_connected_plus_reorder(sentence_split, tail_split):
    return is_tail_in_field_plus_reorder(sentence_split, tail_split, field="conceptnet_related_or_connected")


def is_tail_in_synonyms_plus_reorder(sentence_split, tail_split):
    return is_tail_in_field_plus_reorder(sentence_split, tail_split, field="wordnet_synonyms")
def is_tail_in_hypernyms_plus_reorder(sentence_split, tail_split):
    return is_tail_in_field_plus_reorder(sentence_split, tail_split, field="wordnet_hypernyms")
def is_tail_in_hyponyms_plus_reorder(sentence_split, tail_split):
    return is_tail_in_field_plus_reorder(sentence_split, tail_split, field="wordnet_hyponyms")
def is_tail_in_wordnet_all_plus_reorder(sentence_split, tail_split):
    return is_tail_in_field_plus_reorder(sentence_split, tail_split, field="wordnet_all")


functions_of_interest = [
            is_tail_in_conceptnet_connected_plus_reorder,
            is_tail_in_conceptnet_related_plus_reorder,
            is_tail_in_synonyms_plus_reorder,
            is_tail_in_hypernyms_plus_reorder,
            is_tail_in_hyponyms_plus_reorder,
            is_same_stem_plus_reorder,
        ]

counter = defaultdict(int)
accounted_indexes = defaultdict(list)

name_to_function = OrderedDict()
for func in functions_of_interest:
    func_name = str(func).split()[1]
    name_to_function[func_name] = func

number = 0
unaccounted_indexes = []
for i in trange(len(ground_sentence)):
    sentence = ground_sentence[i]
    tail_entity = tail[i]
    sentence_split = sentence.split()
    tail_split = str(tail_entity).split()

    start_score = 0
    for name in name_to_function:
        increase_in_score = name_to_function[name](sentence_split, tail_split)
        counter[name] += increase_in_score
        start_score += increase_in_score
        if increase_in_score:
            accounted_indexes[name].append(i)

    if start_score == 0:
        unaccounted_indexes.append(i)

unaccounted_pairs = [[ground_sentence[i], tail[i]] for i in unaccounted_indexes]

if analysis_mode != "prediction_analysis":
    print("Total: {}".format(len(ground_sentence)))
    print("Unaccounted samples: {}".format(len(unaccounted_indexes)))
    
    for name in counter:
        print(name, round(counter[name]/len(ground_sentence)*100,2), "%")
    
def save_defaultdict_to_json(data, save_filename):
    with open(save_filename, "w") as write_file:
        json.dump(data, write_file)

if analysis_mode == "prediction_analysis":
    save_defaultdict_to_json(accounted_indexes, "data/genre_accounted_per_category.json")
