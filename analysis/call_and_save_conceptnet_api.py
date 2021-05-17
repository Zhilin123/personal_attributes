#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import pandas as pd
import json
from tqdm import tqdm
import time
import argparse

def unpack_underscore_and_overly_long(entities):
    words = []
    for entity in entities:
        new_words = entity.split("_")
        if len(new_words) < 5:
            words += new_words
    return list(set(words))

def get_words_related_or_connected_to_word(word):
    all_words = get_words_connected_to_word(word) + get_words_related_to_word(word)
    return list(set(all_words))

def get_endpoint(endpoint):
    # minimise effect of rate limiting
    time.sleep(0.75)
    try:
        obj = requests.get(endpoint).json()
    except:
        time.sleep(1)
        obj = requests.get(endpoint).json()
    return obj


def get_words_related_to_word(word):
    related_endpoint = 'http://api.conceptnet.io/related/c/en/{}?filter=/c/en&limit=100'.format(word)
    obj = get_endpoint(related_endpoint)
    related_entities = [i['@id'] for i in obj['related']]
    clean_related_entities = list(set([entity.split('/')[3] for entity in related_entities if len(entity.split('/')) > 2 and entity.split('/')[2] =="en"]))
    all_entities = clean_related_entities
    words_from_entities = unpack_underscore_and_overly_long(all_entities)
    return words_from_entities

def get_words_connected_to_word(word):
    connected_endpoint = 'http://api.conceptnet.io/c/en/{}?filter=/c/en&limit=100'.format(word)
    obj = get_endpoint(connected_endpoint)
    edges = obj['edges']
    ids = [i['@id'] for i in edges]
    connected_entities = []
    for one_id in ids:
        connected_entities += one_id.split(',')[1:]
    clean_connected_entities =list(set([entity.split('/')[3] for entity in connected_entities if len(entity.split('/')) > 2 and entity.split('/')[2] =="en"]))
    words_from_entities = unpack_underscore_and_overly_long(clean_connected_entities)
    return words_from_entities

def get_words(filename, category_of_words="tail"):
    df = pd.read_csv(filename)

    aspect = list(df["ground_{}".format(category_of_words)])
    words = []
    
    for i in aspect:
        words += i.split()
    
    words = list(set(words))
    return words

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='call_conceptnet_api')
    
    parser.add_argument("--field_of_interest", choices=["related","connected"])
    parser.add_argument("--category_of_words", choices=["sentence", "tail"])
    parser.add_argument("--dataset", choices=["train","eval"])
    parser.add_argument("--subset", choices=["all","tail_not_in_sentence"], default="tail_not_in_sentence")
    args = parser.parse_args()
    
    field_of_interest = args.field_of_interest
    category_of_words = args.category_of_words
    dataset = args.dataset
    subset = args.subset
    
    
    filename = "data/{}_sentences_{}.csv".format(dataset, subset)
    output_filename = "data/conceptnet_words/{}_{}_words_to_{}_words_{}.json".format(dataset, category_of_words, field_of_interest,subset)
    
    output_filename.replace("_tail_not_in_sentence","")
    
    field_to_association_func = {
                "related": get_words_related_to_word,
                "connected":get_words_connected_to_word,
                "related_or_connected": get_words_related_or_connected_to_word
            }
    
    words = get_words(filename, category_of_words=category_of_words)
    
    word_to_associated_words = {}
    
    association_func = field_to_association_func[field_of_interest]
    
    for word in tqdm(words):
        word_to_associated_words[word] = association_func(word)
    
    with open(output_filename, "w") as write_file:
        json.dump(word_to_associated_words, write_file)
    