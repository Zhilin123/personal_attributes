#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import spacy
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='preprocess for ling analysis')
parser.add_argument("--csv_filename", help="csv file with eval tokens to generate NER, POS tags and dependency structure")
parser.add_argument("--debug_mode", type=lambda x: (str(x).lower() == 'true'), default=True)
args = parser.parse_args()
csv_filename = args.csv_filename
debug_mode = args.debug_mode

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")

df = pd.read_csv(csv_filename)
df_dict = df.to_dict()

invalid_tokens = ["[HEAD]", "[RELN]", "[TAIL]", '<|startoftext|>', '<|endoftext|>']

for field in ["ground_sentence", "ground_tail", "predicted_tail"]:
    sentences = df_dict[field].values()
    tokens_list = []
    big_pos_tags_list = []
    small_pos_tags_list = []
    dependency_labels_list = []
    ner_list = []
    token_head_text_list = []
    print("Starting with {}".format(field))
    for sentence in tqdm(sentences):
        for invalid_token in invalid_tokens:
            sentence = sentence.replace(invalid_token, '')
        doc = nlp(str(sentence))
        tokens_list.append([token.text for token in doc])
        big_pos_tags_list.append([token.pos_ for token in doc])
        small_pos_tags_list.append([token.tag_ for token in doc])
        dependency_labels_list.append([token.dep_ for token in doc])
        ner_list.append([token.ent_type_ for token in doc])
        token_head_text_list.append([token.head.text for token in doc])
        if debug_mode:
            break

    df_dict["{}_tokens".format(field)] = pd.Series(tokens_list)
    df_dict["{}_big_pos_tags".format(field)] = pd.Series(big_pos_tags_list)
    df_dict["{}_small_pos_tags".format(field)] = pd.Series(small_pos_tags_list)
    df_dict["{}_dependency_labels".format(field)] = pd.Series(dependency_labels_list)
    df_dict["{}_ner".format(field)] = pd.Series(ner_list)
    df_dict["{}_token_head_text".format(field)] = pd.Series(token_head_text_list)

df = pd.DataFrame.from_dict(df_dict)
df.to_csv(csv_filename.replace(".csv", "_analysis.csv"))
