#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

#nayak_filename = "../../nayak_accounted_per_category.json"
genre_filename = "data/genre_accounted_per_category.json"
ground_filename = "data/ground_truth_accounted_per_category.json"

def load_file(filename):
    with open(filename, "r") as read_file:
        data = json.load(read_file)
    return data

ground_file = load_file(ground_filename)

#nayak_file = load_file(nayak_filename)
genre_file = load_file(genre_filename)

predicted_file = genre_file

proportion_overlap = {}

for i in genre_file:
    proportion_overlap[i] = round(100*len(set(ground_file[i]).intersection(set(predicted_file[i]))) / len(ground_file[i]),1)
    
keys = [
        'is_tail_in_conceptnet_related_plus_reorder',
        'is_tail_in_conceptnet_connected_plus_reorder',
        'is_tail_in_synonyms_plus_reorder',
        'is_tail_in_hypernyms_plus_reorder',
        'is_tail_in_hyponyms_plus_reorder',
        'is_same_stem_plus_reorder'
        ]

for key in keys:
    print(key, proportion_overlap[key])





