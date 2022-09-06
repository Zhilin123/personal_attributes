#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import torch
import numpy as np
import csv
import copy
from tqdm import tqdm, trange
import math
import os
import time
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoTokenizer
from generator_dataset import DNLIDataset
from torch import nn
import argparse
from sklearn import metrics
from collections import defaultdict

parser = argparse.ArgumentParser(description='train_model')
parser.add_argument("--debug_mode", type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument("--load_trained_model", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--need_generation", type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument("--train_generation", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--lr", default="1e-4") #5e-5
parser.add_argument("--warmup_steps", default="1e2")
parser.add_argument("--config_name", default="default")
parser.add_argument("--inference_only", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Only performs generation without further training; load_trained model must be True")
parser.add_argument("--generation_name", default="num_beams-1",
                    help="top_k-5, top_p-0.5, temperature-0.5, num_beams-5, original_tokens, original_spans, only_valid,  commonsense_tokens, only_original_valid")
parser.add_argument("--data_subset", default="all", choices=["all", "within_sentence", "not_within_sentence"])
parser.add_argument("--mode", default="head_reln_tail",choices=[
                                            "head_reln_tail", "head_tail_reln",
                                            "reln_head_tail", "reln_tail_head",
                                            "tail_reln_head", "tail_head_reln",
                                            "head", "reln", "tail"]) # dropped support for all_together; generation_name only supports head_reln_tail, head, reln, tail

parser.add_argument("--model_name", default="gpt2", help="gpt2, dialogpt, gpt2-medium")
parser.add_argument("--generate_train", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--generate_test", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--generate_one_batch_only", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--generate_custom", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--generate_json_filename", default="", help="")
parser.add_argument("--unified_ontology", type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument("--random_seed", type=int, default=42)

args = parser.parse_args()

debug_mode = args.debug_mode
load_trained_model = args.load_trained_model
epochs = args.epochs
max_epochs = args.max_epochs
need_generation = args.need_generation
train_generation = args.train_generation
learning_rate = float(args.lr) #5e-5 #5e-4 #
warmup_steps = float(args.warmup_steps)
config_name = args.config_name
inference_only = args.inference_only
generation_name = args.generation_name
data_subset = args.data_subset
mode = args.mode #"all_together" #
model_name = args.model_name
generate_train = args.generate_train
generate_test = args.generate_test
generate_one_batch_only = args.generate_one_batch_only
generate_json_filename = args.generate_json_filename
generate_custom = args.generate_custom
unified_ontology = args.unified_ontology

print("load_trained_model: ", load_trained_model)
print("need_generation: ", need_generation)
print("lr: ", learning_rate)
print("config_name: ", config_name)
print("mode: ", mode)
print("data_subset: ", data_subset)

if len(generation_name):
    print("generation_name: ", generation_name)
#if not train_generation:
#    print("please do not rely on training avg_correct metrics; skipping generation during training to save time")
if "_" not in mode or mode == "all_together":
    print("please only look at the section for intended label; for all_together, \
          the metric cannot be automatically calculated now. please do eval afterwards")

# Set the seed value all over the place to make this reproducible.
seed_val = args.random_seed

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

sample_every = 400 if not debug_mode else 1


if not os.path.exists(config_name+"/"):
    os.makedirs(config_name+"/")

checkpointer_name = "{}/pytorch_model.pth".format(config_name)
best_checkpointer_name = "{}/pytorch_model_best.pth".format(config_name)
training_stats_filename = "{}/training_stats.csv".format(config_name)
eval_stats_filename = "{}/eval_stats.csv".format(config_name)


if inference_only and model_name == "gpt2-medium":
    batch_size = 2
elif model_name == "gpt2-medium":
    batch_size = 8
else:
    batch_size = 16


eval_every = 0.25

epsilon = 1e-8

adjust_sample_weight = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if "gpt2" in model_name:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name) #, add_prefix_space=True
    configuration = GPT2Config.from_pretrained(model_name, output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)

elif model_name == "dialogpt":
    print("using dialogpt")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

tokenizer.padding_side = "right" # "left"

if not unified_ontology:
    relations = ['[attend_school]','[dislike]','[employed_by_company]',
                 '[employed_by_general]','[favorite]','[favorite_activity]',
                 '[favorite_animal]','[favorite_book]','[favorite_color]',
                 '[favorite_drink]','[favorite_food]','[favorite_hobby]',
                 '[favorite_movie]','[favorite_music]','[favorite_music_artist]',
                 '[favorite_place]','[favorite_season]','[favorite_show]',
                 '[favorite_sport]','[gender]','[has_ability]','[has_age]',
                 '[has_degree]','[has_hobby]','[has_profession]','[have]',
                 '[have_chidren]','[have_family]','[have_pet]','[have_sibling]',
                 '[have_vehicle]','[job_status]','[like_activity]','[like_animal]',
                 '[like_drink]','[like_food]','[like_general]','[like_goto]',
                 '[like_movie]','[like_music]','[like_read]','[like_sports]',
                 '[like_watching]','[live_in_citystatecountry]','[live_in_general]',
                 '[marital_status]','[member_of]','[misc_attribute]','[nationality]',
                 '[not_have]','[other]','[own]','[physical_attribute]','[place_origin]',
                 '[previous_profession]','[school_status]','[teach]',
                 '[want]','[want_do]','[want_job]']
else:
    relations = ['[attend_school]', '[employed_by_company]', '[employed_by_general]',
                 '[favorite_color]', '[favorite_music_artist]','[favorite_season]',
                 '[gender]', '[has_ability]', '[has_age]', '[has_degree]',
                 '[has_profession]', '[have_family]','[have_pet]', '[have_vehicle]',
                 '[job_status]','[like_activity]', '[like_animal]', '[like_drink]',
                 '[like_food]', '[like_goto]', '[like_movie]', '[like_music]',
                 '[like_read]', '[like_sports]', '[like_watching]',
                 '[live_in_citystatecountry]', '[live_in_general]',
                 '[marital_status]', '[member_of]', '[misc_attribute]',
                 '[nationality]', '[own]', '[physical_attribute]',
                 '[place_origin]', '[previous_profession]', '[school_status]',
                 '[teach]', '[want_do]', '[want_job]']




tokenizer.add_special_tokens({'pad_token': '<|endoftext|>',
                              'bos_token': '<|startoftext|>',
                              'eos_token': '<|endoftext|>',
                              "additional_special_tokens":["[HEAD]", "[RELN]", "[TAIL]"] + relations
                              })

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
model.resize_token_embeddings(len(tokenizer))

optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )



def load_conceptnet(conceptnet_filename):
    # to account for when word is not in data
    word_to_conceptnet_associated_words = defaultdict(list)

    with open(conceptnet_filename, "r") as read_file:
       data = json.load(read_file)

    for i in data:
        word_to_conceptnet_associated_words[i] = data[i]

    return word_to_conceptnet_associated_words

def combine(defaultdict0, defaultdict1):
    new_defaultdict = defaultdict(list)
    all_keys = list(set(list(defaultdict0.keys()) + list(defaultdict1.keys())))
    for i in all_keys:
        new_defaultdict[i] = list(set( defaultdict0[i] + defaultdict1[i])) #[i] +
    return new_defaultdict

def remove_duplicate(one_default_dict):
    new_default_dict = defaultdict(list)

    for key in one_default_dict:
        new_default_dict[key] = list(set(one_default_dict[key]))

    return new_default_dict

def map_tokens_to_G_tokens():
    token_id_to_g_token_id = defaultdict(list)
    g_token_id_to_token_id = defaultdict(list)

    for token_id in all_tokens_except_special_relation:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if len(token.strip())> 0:
            g_token = " " +token
            # g_token_id is can be int or list
            g_token_id = tokenizer.encode(g_token)
            if isinstance(g_token_id, int):
                token_id_to_g_token_id[token_id].append(g_token_id)
                #g_token_id_to_token_id[g_token_id].append(token_id)
            else:
                token_id_to_g_token_id[token_id] += g_token_id
#                for one_g_token_id in g_token_id:
#                    g_token_id_to_token_id[one_g_token_id].append(token_id)
#
    for token_id in all_tokens_except_special_relation:
        token = tokenizer.convert_ids_to_tokens(token_id)
        token_without_g = token.replace("Ġ","")
        if len(token_without_g) < len(token):
            token_id_without_g = tokenizer.encode(token_without_g)
            if isinstance(token_id_without_g, int):
                g_token_id_to_token_id[token_id].append(token_id_without_g)
            else:
                g_token_id_to_token_id[token_id] += token_id_without_g


    return remove_duplicate(token_id_to_g_token_id), remove_duplicate(g_token_id_to_token_id)



if load_trained_model:
    #model_state_dict = torch.load(bin_name)
    #model.load_state_dict(model_state_dict)
    # load
    if inference_only:
        checkpoint = torch.load(best_checkpointer_name, map_location=device)
    else:
        checkpoint = torch.load(checkpointer_name, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    last_finished_epoch = checkpoint['epoch']
    starting_epoch = last_finished_epoch + 1

    with open(training_stats_filename) as f:
        training_stats = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]

    print("starting_epoch: ", starting_epoch)



else:
    starting_epoch = 0
    training_stats = []


model = model.to(device)

if inference_only and "associated_with_relation" not in generation_name:
    train_dataset = DNLIDataset("dnli/dialogue_nli/dialogue_nli_train.jsonl" ,
                                tokenizer, debug_mode=True, data_subset=data_subset, mode=mode,
                                unified_ontology=unified_ontology)
else:
    train_dataset = DNLIDataset("dnli/dialogue_nli/dialogue_nli_train.jsonl" ,
                                tokenizer, debug_mode=debug_mode, data_subset=data_subset, mode=mode,
                                unified_ontology=unified_ontology)

if generate_train:
    val_dataset = DNLIDataset("dnli/dialogue_nli/dialogue_nli_train.jsonl",
                          tokenizer, debug_mode=debug_mode, data_subset=data_subset, mode=mode,
                          unified_ontology=unified_ontology)
elif generate_test:
    val_dataset = DNLIDataset("dnli/dialogue_nli/dialogue_nli_test.jsonl",
                          tokenizer, debug_mode=debug_mode, data_subset=data_subset, mode=mode,
                          unified_ontology=unified_ontology)
elif generate_custom:
    val_dataset = DNLIDataset(generate_json_filename,
                          tokenizer, debug_mode=debug_mode, data_subset=data_subset, mode=mode,
                          unified_ontology=unified_ontology)
else:
    val_dataset = DNLIDataset("dnli/dialogue_nli/dialogue_nli_dev.jsonl",
                          tokenizer, debug_mode=debug_mode, data_subset=data_subset, mode=mode,
                          unified_ontology=unified_ontology)

print('{:>5,} training samples'.format(len(train_dataset)))
print('{:>5,} validation samples'.format(len(val_dataset)))


train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )


validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
        )

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = warmup_steps,
                                            num_training_steps = total_steps)

if load_trained_model:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# save
def save(model, optimizer, scheduler,checkpointer_name, epoch):
    output_dir = "/".join(checkpointer_name.split("/")[:-1]) + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model
    # save
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict':scheduler.state_dict(),
        'epoch':epoch
    }, checkpointer_name)


id_of_special_tokens = tokenizer.convert_tokens_to_ids(["[HEAD]", "[RELN]", "[TAIL]"] +
                                                       ['<|startoftext|>', '<|endoftext|>'] +
                                                       relations)

head_token_id = tokenizer.convert_tokens_to_ids("[HEAD]")
reln_token_id = tokenizer.convert_tokens_to_ids("[RELN]")
tail_token_id = tokenizer.convert_tokens_to_ids("[TAIL]")
eos_token_id = tokenizer.eos_token_id
bos_token_id = tokenizer.bos_token_id

special_relation_token_ids_set = set(tokenizer.convert_tokens_to_ids(relations))
all_tokens_except_special_relation = [i for i in range(len(tokenizer)) if i not in id_of_special_tokens]
all_tokens = [i for i in range(len(tokenizer))]
token_id_to_g_token_id, g_token_id_to_token_id = map_tokens_to_G_tokens()
token_id_to_g_token_id_vice_versa = combine(token_id_to_g_token_id, g_token_id_to_token_id)




def get_tokens_to_conceptnet_tokens(words_to_conceptnet_words, tokenizer):
    tokens_to_conceptnet_tokens = defaultdict(list)
    for word in tqdm(words_to_conceptnet_words):
        key_tokens = set(tokenizer.encode(word))
        value_tokens = []
        for i in words_to_conceptnet_words[word]:
            value_tokens += tokenizer.encode(i)
        value_tokens = list(set(value_tokens))
        for key_token in key_tokens:
            tokens_to_conceptnet_tokens[key_token] += value_tokens
    non_repeat_tokens_to_conceptnet_tokens = defaultdict(list)
    for key in tokens_to_conceptnet_tokens:
        non_repeat_tokens_to_conceptnet_tokens[key] = list(set(tokens_to_conceptnet_tokens[key]))
    return non_repeat_tokens_to_conceptnet_tokens

def get_relation_token_to_tail_span(all_input_ids):
    # does not consider g_token issue
    relation_token_to_tail_span = defaultdict(list)
    relation_token_to_tail_token = defaultdict(list)
    for input_id in all_input_ids:
        all_tokens = [i.item() for i in input_id]
        reln_token = [i for i in all_tokens if i in special_relation_token_ids_set][0]
        tail_token_position = [i for i in range(len(all_tokens)) if all_tokens[i] == tail_token_id][0]
        tail_tokens = [i for i in all_tokens[tail_token_position+1:] if i != eos_token_id]
        relation_token_to_tail_span[reln_token].append(tuple(tail_tokens))
        relation_token_to_tail_token[reln_token] += tail_tokens

    return remove_duplicate(relation_token_to_tail_span), remove_duplicate(relation_token_to_tail_token)





if generation_name == "commonsense_tokens":
    word_to_conceptnet_connected_words_filename = "conceptnet_words/eval_sentence_words_to_connected_words_all.json"
    word_to_conceptnet_related_words_filename = "conceptnet_words/eval_sentence_words_to_related_words_all.json"
    words_to_conceptnet_words = combine(load_conceptnet(word_to_conceptnet_connected_words_filename),
                                         load_conceptnet(word_to_conceptnet_related_words_filename))
    tokens_to_conceptnet_tokens = get_tokens_to_conceptnet_tokens(words_to_conceptnet_words, tokenizer)

if "associated_with_relation" in generation_name:
    relation_token_to_tail_span, relation_token_to_tail_token = get_relation_token_to_tail_span(train_dataset.input_ids)


#def get_bad_word_ids(b_generate_input_ids):
#    #all_bad_word_ids = []
#    for i in range(len(b_generate_input_ids)):
#        token_ids_in_generate_text = [j.item() for j in b_generate_input_ids[i]]
#        set_token_ids_in_generate_text = list(set(token_ids_in_generate_text))
#        permitted_ids = set(set_token_ids_in_generate_text + id_of_special_tokens)
#        bad_word_ids = [[i] for i in range(len(tokenizer)) if i not in permitted_ids]
#        #all_bad_word_ids.append(bad_word_ids)
##        print(len(b_generate_input_ids))
##        print(permitted_ids)
##        print(sorted(tokenizer.convert_ids_to_tokens(b_generate_input_ids[i])))
##        print(len(sorted(tokenizer.convert_ids_to_tokens(list(permitted_ids)))))
##        print(len(bad_word_ids), len(tokenizer))
#        #raise ValueError
#
#    return bad_word_ids

#def remove_tokens_before_bos(one_input_ids):
#    all_tokens = [i.item() for i in one_input_ids] # #.cpu().detach().numpy()
#    location_of_bos = [i for i in range(len(all_tokens)) if all_tokens[i] == tokenizer.bos_token_id][0]
#    return all_tokens[location_of_bos:]

#def get_candidate_head_and_tail(candidate):
#    location_of_special_tokens = [(i, candidate[i]) for i in range(len(candidate)) if candidate[i] in [head_token_id, reln_token_id, tail_token_id, eos_token_id]]
#    head_tokens = None
#    tail_tokens = None
#    for i in range(len(location_of_special_tokens)):
#        index, token_id = location_of_special_tokens[i]
#        if token_id == head_token_id and i < len(location_of_special_tokens)-1:
#            head_tokens = candidate[index+1:location_of_special_tokens[i+1][0]]
#        if token_id == tail_token_id and i < len(location_of_special_tokens)-1:
#            tail_tokens = candidate[index+1:location_of_special_tokens[i+1][0]]
#    return head_tokens, tail_tokens


#def check_if_token_in_list(input_ids, span, filter_by="tokens"):
#    if filter_by=="tokens":
#        for token in span:
#            if token not in input_ids:
#                return False
#        return True
#    # in the case of filter by spans
#    else:
#        for i in range(len(input_ids)-len(span)):
#            if span == input_ids[i:i+len(span)]:
#                return True
#        return False

#def check_if_both_head_and_tail_in_original_sentence_span(input_ids, candidate):
#    head_tokens, tail_tokens = get_candidate_head_and_tail(candidate)
#    if head_tokens is None or tail_tokens is None or len(head_tokens) < 1 or len(tail_tokens) < 1:
#        return False
#    if check_if_span_in_list(input_ids, head_tokens) and check_if_span_in_list(input_ids, tail_tokens):
#        return True
#    return False

#
#def filter_all_possible_sequences_by_spans(b_generate_input_ids, all_possible_return_seq):
#    all_optimal_answers = []
#    for i in range(len(b_generate_input_ids)):
#        one_input_ids = b_generate_input_ids[i]
#        # this is now a list
#        one_input_ids = remove_tokens_before_bos(one_input_ids)
#        candidates = all_possible_return_seq[i*10:(i+1)*10]
#        optimal_answer = candidates[0]
#        for j in range(len(candidates)):
#            candidate = candidates[j]
#            candidate = remove_tokens_before_bos(candidate)
#            if check_if_both_head_and_tail_in_original_sentence_span(one_input_ids, candidate):
#                optimal_answer = candidates[j]
#                break
#        all_optimal_answers.append(optimal_answer)
#
#    return torch.stack(all_optimal_answers)

#def check_if_both_head_and_tail_in_original_sentence_token(input_ids, candidate, filter_by="tokens"):
#    head_tokens, tail_tokens = get_candidate_head_and_tail(candidate)
#    if head_tokens is None or tail_tokens is None or len(head_tokens) < 1 or len(tail_tokens) < 1:
#        return False
#    if check_if_token_in_list(input_ids, head_tokens, filter_by=filter_by) and check_if_token_in_list(input_ids, tail_tokens,filter_by=filter_by):
#        return True
#    return False

#def filter_all_possible_sequences_by_tokens(b_generate_input_ids, all_possible_return_seq, filter_by="tokens"):
#    all_optimal_answers = []
#    for i in range(len(b_generate_input_ids)):
#        one_input_ids = b_generate_input_ids[i]
#        # this is now a list
#        one_input_ids = remove_tokens_before_bos(one_input_ids)
#        candidates = all_possible_return_seq[i*10:(i+1)*10]
#        #print(candidates)
#        #raise ValueError
#        optimal_answer = candidates[0]
#        for j in range(len(candidates)):
#            candidate = candidates[j]
#            candidate = remove_tokens_before_bos(candidate)
#            if filter_by in ["tokens","spans"]:
#                condition = check_if_both_head_and_tail_in_original_sentence_token(one_input_ids, candidate, filter_by=filter_by)
#            else:
#                # if filter_by = "valid"
#                condition = validate_predicted_tokens(candidate)
#            if condition:
#                optimal_answer = candidates[j]
#                break
#        all_optimal_answers.append(optimal_answer)
#
#    return torch.stack(all_optimal_answers)

#def only_allow_tokens_from_sentence(batch_id, input_ids):
#    token_ids_in_generate_text = [j.item() for j in input_ids]
#    set_token_ids_in_generate_text = list(set(token_ids_in_generate_text))
#    permitted_ids = list(set(set_token_ids_in_generate_text + id_of_special_tokens))
#    return permitted_ids
#
#def only_allow_spans_from_sentence(batch_id, input_ids):
#    token_ids_in_generate_text = [j.item() for j in input_ids]
#    last_token_id = token_ids_in_generate_text[-1]
#    accepted_tokens = []
#    for i in range(len(token_ids_in_generate_text)-1):
#        if token_ids_in_generate_text[i] == last_token_id:
#            accepted_tokens.append(token_ids_in_generate_text[i+1])
#    permitted_ids = list(set(accepted_tokens + id_of_special_tokens))
#    return permitted_ids
#
#def only_allow_valid_tokens(batch_id, input_ids):
#    # this function can be used in combination with other filtering functions
#    last_token_id = input_ids[-1].item()
#
#    #after special relation tokens only allow [TAIL]
#    if last_token_id in special_relation_token_ids_set:
#        return [tail_token_id] if len(series) == 4 else [eos_token_id]
#
#    #after [RELN] only allow the special relation tokens
#    elif last_token_id == reln_token_id:
#        return list(special_relation_token_ids_set)
#
#    #after [HEAD] and [TAIL] only allow strings
#    elif last_token_id == head_token_id or last_token_id == tail_token_id:
#        return all_tokens_except_special_relation + [eos_token_id]
#    # normal string tokens
#    else:
#        return all_tokens

def get_alternate_g_or_no_g_tokens(permitted_ids):
    permitted_ids_considering_g_tokens = []

    for permitted_id in permitted_ids:
        permitted_ids_considering_g_tokens.append(permitted_id)
        permitted_ids_considering_g_tokens += token_id_to_g_token_id_vice_versa[permitted_id]

    return list(set(permitted_ids_considering_g_tokens))

## these are for Extraction dataset

def only_allow_valid_tokens_from_sentence(batch_id, input_ids):
    return constraint_generation(batch_id, input_ids, mode="original_tokens")

def only_allow_valid_spans_from_sentence(batch_id, input_ids):
    return constraint_generation(batch_id, input_ids, mode="original_spans")

def only_allow_tail_tokens_associated_with_relation_from_sentence(batch_id, input_ids):
    tokens_from_sentence = constraint_generation(batch_id, input_ids, mode="original_tokens")
    tokens_associated_with_relation = constraint_generation(batch_id, input_ids, mode="tail_tokens_associated_with_relation")
    intersection_tokens = set(tokens_from_sentence).intersection(set(tokens_associated_with_relation))
    return list(intersection_tokens)

def only_allow_tail_spans_associated_with_relation_from_sentence(batch_id, input_ids):
    tokens_from_sentence = constraint_generation(batch_id, input_ids, mode="original_spans")
    tokens_associated_with_relation = constraint_generation(batch_id, input_ids, mode="tail_spans_associated_with_relation")
    intersection_tokens = set(tokens_from_sentence).intersection(set(tokens_associated_with_relation))
    return list(intersection_tokens)

## These are for inference dataset

def only_allow_tokens_and_conceptnet_from_sentence(batch_id, input_ids):
    return constraint_generation(batch_id, input_ids, mode="commonsense_tokens")

def only_allow_valid_tokens(batch_id, input_ids):
    return constraint_generation(batch_id, input_ids, mode="only_valid")

def only_allow_tail_tokens_associated_with_relation(batch_id, input_ids):
    return constraint_generation(batch_id, input_ids, mode="tail_tokens_associated_with_relation")

def only_allow_tail_spans_associated_with_relation(batch_id, input_ids):
    return constraint_generation(batch_id, input_ids, mode="tail_spans_associated_with_relation")

def only_allow_tail_tokens_associated_with_relation_and_commonsense(batch_id, input_ids):
    tokens_from_sentence = constraint_generation(batch_id, input_ids, mode="commonsense_tokens")
    tokens_associated_with_relation = constraint_generation(batch_id, input_ids, mode="tail_tokens_associated_with_relation")
    intersection_tokens = set(tokens_from_sentence).intersection(set(tokens_associated_with_relation))
    return list(intersection_tokens)

def only_allow_tail_spans_associated_with_relation_and_commonsense(batch_id, input_ids):
    tokens_from_sentence = constraint_generation(batch_id, input_ids, mode="commonsense_tokens")
    tokens_associated_with_relation = constraint_generation(batch_id, input_ids, mode="tail_spans_associated_with_relation")
    intersection_tokens = set(tokens_from_sentence).intersection(set(tokens_associated_with_relation))
    return list(intersection_tokens)

'''
Test cases

edris island	islandedris	that is awesome! i live on an island edris island.
jujitsu	ju	not sure yet. i am learning jujitsu too but its still quite hard.
sardines	s	i eat sardines for breakfast.
nirvana	n	i love nirvana a lot.
california	n yc	california, i grew up here but i am moving to nyc next year.
cat	t	i took in a stray cat. i fed him once and he never left.
'''
def get_candidate_tokens(token_ids_in_generate_text, last_token_id, mode="original_tokens"):

    if mode == "original_tokens":
        original_tokens = token_ids_in_generate_text
        candidate_tokens = []
        for original_token in original_tokens:
            candidate_tokens += token_id_to_g_token_id_vice_versa[original_token]
            candidate_tokens.append(original_token)

    elif mode == "original_spans":

        variants_of_last_token_id = [last_token_id] + token_id_to_g_token_id_vice_versa[last_token_id]

        for token in token_ids_in_generate_text:
            g_tokens = token_id_to_g_token_id_vice_versa[token]
            if last_token_id in g_tokens:
                variants_of_last_token_id.append(token)

        tokens_following_last_token = [token_ids_in_generate_text[i+1]
                                        for i in range(len(token_ids_in_generate_text)-1)
                                            if token_ids_in_generate_text[i] in variants_of_last_token_id]


#        tokens_following_last_token += [token_ids_in_generate_text[i]
#                                        for i in range(len(token_ids_in_generate_text))
#                                            if token_ids_in_generate_text[i] in variants_of_last_token_id]

        for token in token_ids_in_generate_text:
            g_tokens = token_id_to_g_token_id_vice_versa[token]
            if last_token_id in g_tokens:
                tokens_following_last_token += g_tokens

        candidate_tokens = [i for i in tokens_following_last_token if i not in id_of_special_tokens]


    elif mode == "commonsense_tokens":

        variants_of_token_ids_in_generate_text = get_alternate_g_or_no_g_tokens(token_ids_in_generate_text)

        variants_of_token_ids_in_generate_text_including_conceptnet = []

        for token_id in variants_of_token_ids_in_generate_text:
            variants_of_token_ids_in_generate_text_including_conceptnet.append(token_id)
            variants_of_token_ids_in_generate_text_including_conceptnet += tokens_to_conceptnet_tokens[token_id]

        candidate_tokens = list(set(variants_of_token_ids_in_generate_text_including_conceptnet))

    elif mode == "only_valid":
        candidate_tokens = all_tokens_except_special_relation

    elif mode == "tail_tokens_associated_with_relation":
        # [TAIL] hasn't appeared
        if tail_token_id not in token_ids_in_generate_text:
            return get_candidate_tokens(token_ids_in_generate_text, last_token_id, mode="only_valid")
        else:
            candidate_tokens = []
            reln_token = [i for i in token_ids_in_generate_text if i in special_relation_token_ids_set][0]
            candidate_tokens = relation_token_to_tail_token[reln_token]

    elif mode == "tail_spans_associated_with_relation":
        # [TAIL] hasn't appeared
        if tail_token_id not in token_ids_in_generate_text:
            return get_candidate_tokens(token_ids_in_generate_text, last_token_id, mode="only_valid")
        else:
            reln_token = [i for i in token_ids_in_generate_text if i in special_relation_token_ids_set][0]
            candidate_spans = relation_token_to_tail_span[reln_token]

            variants_of_last_token_id = [last_token_id] + token_id_to_g_token_id_vice_versa[last_token_id]

            for token in token_ids_in_generate_text:
                # this is to account for word fragments that might have been split differently
                # blonde vs bl onde (so if bl is the last token id, make sure ondo is also part of next possible)
                g_tokens = token_id_to_g_token_id_vice_versa[token]
                if last_token_id in g_tokens:
                    variants_of_last_token_id.append(token)

            tokens_following_last_token = []

            for variant_of_last_token_id in variants_of_last_token_id:
                for candidate_span in candidate_spans:
                    if variant_of_last_token_id in candidate_span:
                        tokens_following_last_token += list(candidate_span)

            candidate_tokens = tokens_following_last_token

    return candidate_tokens


def constraint_generation(batch_id, input_ids, mode="original_spans"):

    last_token_id = input_ids[-1].item()

    #after special relation tokens only allow [TAIL]
    if last_token_id in special_relation_token_ids_set:
        permitted_ids = [tail_token_id] if len(series) == 4 else [eos_token_id]

    #after [RELN] only allow the special relation tokens
    elif last_token_id == reln_token_id:
        permitted_ids = list(special_relation_token_ids_set)

    # when generated eos, the only possibility is other eos
    elif last_token_id == eos_token_id:
        permitted_ids = [eos_token_id]

    #after [HEAD] and [TAIL] only allow strings
    elif last_token_id == head_token_id or last_token_id == tail_token_id:
        token_ids_in_generate_text = [j.item() for j in input_ids]
        # When generating after [HEAD] or [TAIL], span constraint should not apply
        head_tail_mode = mode.replace("spans", "tokens")

        # for the head field, tail_X_associated_with_relation does not apply
        # therefore swap out for basic version
        # Note: tail_spans_associated_with_relation is covered because of above
        if last_token_id == head_token_id:
            head_tail_mode = head_tail_mode.replace("tail_tokens_associated_with_relation", "only_valid")
            head_tail_mode = head_tail_mode.replace("commonsense_tokens", "only_valid")


        candidate_tokens = get_candidate_tokens(token_ids_in_generate_text, last_token_id, mode=head_tail_mode)
        set_of_output = set(candidate_tokens)
        special_tokens = series + [bos_token_id]
        for special_token in special_tokens:
            set_of_output.discard(special_token)
        permitted_ids = list(set_of_output)

    # after normal string tokens
    else:
        token_ids_in_generate_text = [j.item() for j in input_ids]

        # tail_X_associated_with_relation does not apply when [TAIL] has not been generated
        if tail_token_id not in token_ids_in_generate_text:
            mode = mode.replace("tail_tokens_associated_with_relation", "only_valid")
            mode = mode.replace("tail_spans_associated_with_relation", "only_valid")
            mode = mode.replace("commonsense_tokens", "only_valid")

        candidate_tokens = get_candidate_tokens(token_ids_in_generate_text, last_token_id, mode=mode)

        if len(series) == 2:
            permitted_ids = [eos_token_id] + candidate_tokens
        elif len(series) == 4:
            if reln_token_id not in token_ids_in_generate_text:
                permitted_ids = [reln_token_id] + candidate_tokens
            elif tail_token_id not in token_ids_in_generate_text:
                permitted_ids = [tail_token_id] + candidate_tokens
            else:
                permitted_ids = [eos_token_id] + candidate_tokens

        #if len(series) == 4
#    print(head_token_id)
#    print(batch_id)


#    print(last_token_id)
#    print("eos", eos_token_id)
#    print(permitted_ids)
#    print(input_ids)
#    print(tokenizer.convert_ids_to_tokens(input_ids))
#    print(tokenizer.convert_ids_to_tokens(permitted_ids))
#


    if mode != "only_valid":
        # this is to account for G tokens due to GPT2 tokenization meaning
        # that the same word can be tokenized in two different ways
        permitted_ids = get_alternate_g_or_no_g_tokens(permitted_ids)

    #print(tokenizer.convert_ids_to_tokens(permitted_ids_considering_g_tokens))
    #raise ValueError

    return permitted_ids




"""
prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
If provided, this function constraints the beam search to allowed tokens only at each step. If not
provided no constraint is applied. This function takes 2 arguments :obj:`inputs_ids` and the batch ID
:obj:`batch_id`. It has to return a list with the allowed tokens for the next generation step
conditioned on the previously generated tokens :obj:`inputs_ids` and the batch ID :obj:`batch_id`. This
argument is useful for constrained generation conditioned on the prefix, as described in
`Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.
"""

# re-implement original_tokens

def generate(b_generate_input_ids, b_generate_attn_masks):
    generated_tokens = []
    sequences_scores = []
    max_length = 128
    for i in range(b_generate_input_ids.size(0)):
        template_length = torch.sum(b_generate_attn_masks[i : i + 1])
        single_generated_sequence, single_sequences_scores = single_generate(b_generate_input_ids[i : i + 1, : template_length], b_generate_attn_masks[i : i + 1, : template_length])
        generated_tokens.append(single_generated_sequence)
        sequences_scores.append(single_sequences_scores)

    # pad each generated to ensure they are of same length in dim 1
    generated_tokens = [
        torch.cat(
            [i, torch.ones((i.size(0), max_length - i.size(1))).to(i.device) * tokenizer.pad_token_id],
            axis=-1,
        )
        for i in generated_tokens
    ]
    generated_tokens = torch.cat(generated_tokens, axis=0)
    return generated_tokens, torch.cat(sequences_scores, axis=0)

def single_generate(b_generate_input_ids, b_generate_attn_masks):


    param_dict = {
              "input_ids": b_generate_input_ids,
              "attention_mask": b_generate_attn_masks,
              "pad_token_id": tokenizer.pad_token_id,
              "no_repeat_ngram_size": 2, # no repeating of 2-grams
              "max_length": 128,
              "output_scores": True,
              "return_dict_in_generate": True
            }

    #repetition_penalty=None,
    #temperature=None,
    #bad_word_ids=
#    if generation_name == "original_tokens":
#        #param_dict["bad_words_ids"] = get_bad_word_ids(b_generate_input_ids)
#        param_dict["num_beams"] = 10
#        param_dict["num_return_sequences"] = 10

    gen_name_to_constraint_func = {
            "original_spans":only_allow_valid_spans_from_sentence,
            "original_tokens":only_allow_valid_tokens_from_sentence,
            "only_valid": only_allow_valid_tokens,
            "commonsense_tokens":only_allow_tokens_and_conceptnet_from_sentence,
            "tail_tokens_associated_with_relation":only_allow_tail_tokens_associated_with_relation,
            "tail_spans_associated_with_relation":only_allow_tail_spans_associated_with_relation,
            "original_tokens-tail_tokens_associated_with_relation":only_allow_tail_tokens_associated_with_relation_from_sentence,
            "original_spans-tail_spans_associated_with_relation":only_allow_tail_spans_associated_with_relation_from_sentence,
            "commonsense_tokens-tail_tokens_associated_with_relation":only_allow_tail_tokens_associated_with_relation_from_sentence,
            "commonsense_tokens-tail_spans_associated_with_relation":only_allow_tail_spans_associated_with_relation_from_sentence,
            }

    if generation_name.replace("+all_ten","") in gen_name_to_constraint_func:
        param_dict["num_beams"] = 10

    if "+all_two" in generation_name:
        param_dict["num_beams"] = 10
        param_dict["num_return_sequences"] = 2
        if '+' in generation_name:
            param_dict["prefix_allowed_tokens_fn"] = gen_name_to_constraint_func[generation_name.split("+")[0]]


    elif "+all_ten" in generation_name or "first" in generation_name.split("-")[0]:
        #param_dict["bad_words_ids"] = get_bad_word_ids(b_generate_input_ids)
        param_dict["prefix_allowed_tokens_fn"] = gen_name_to_constraint_func[generation_name.split("+")[0]]
        param_dict["num_beams"] = 10
        param_dict["num_return_sequences"] = 10

    else:
        param_name, param_val = generation_name.split("-")
        param_val = float(param_val)
        if param_val >= 1:
            param_val = int(param_val)
        param_dict[param_name] = param_val

        if param_name in ["top_k", "top_p", "temperature"]:
            param_dict["do_sample"] = True

    if "first" not in generation_name.split("-")[0]:
        model_output = model.generate(**param_dict)
        all_possible_return_seq = model_output.sequences
        sequences_scores = model_output.sequences_scores
        return all_possible_return_seq, sequences_scores
    else:
        # shape is [(batch*num_return_sequence) x max_seq_len]
        model_output = model.generate(**param_dict)
        all_possible_return_seq = model_output.sequences
        sequences_scores = model_output.sequences_scores
        index = int(generation_name.split("-")[1])-1
        return torch.stack([all_possible_return_seq[index]]), torch.stack([sequences_scores[index]])
        #return torch.stack([all_possible_return_seq[one_index] for one_index in range(int(generation_name.split("-")[1])-1, len(all_possible_return_seq), 10)])

        #if generation_name.split("-")[0] == "first":
            #print(all_possible_return_seq.size())
            #print(torch.stack([all_possible_return_seq[one_index] for one_index in range(int(generation_name.split("-")[1])-1, len(all_possible_return_seq), 10)]).size())
            #return filter_all_possible_sequences_by_tokens(b_generate_input_ids, all_possible_return_seq, filter_by=generation_name.split("_")[1])
#        elif generation_name == "original_spans":
#            return filter_all_possible_sequences_by_spans(b_generate_input_ids, all_possible_return_seq)



def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

series = tokenizer.convert_tokens_to_ids(["[{}]".format(i) for i in mode.upper().split("_")]) + [eos_token_id]

def exclude_special_tokens(tensor1d):
    return [i for i in tensor1d if i not in [head_token_id, reln_token_id, tail_token_id]]

def get_start_and_end_token_ids(category="head", mode="head_reln_tail"):
    if mode != "all_together":
        mode_split = mode.split("_")
        for i in range(len(mode_split)):
            if category == mode_split[i]:
                start_token_id = series[i]
                end_token_id = series[i+1]
    else:
        category_to_start_token_id = {
                    "head":0,
                    "reln":1,
                    "tail":2
                }
        start_token_id = series[category_to_start_token_id[category]]
        end_token_id = series[-1]

    return start_token_id, end_token_id

def get_span(ground_truth_tokens, predicted_tokens, category="head"):
    # for situations in which the true output only has one special token
    #(ie head, tail or reln ) instead of all three
    if category not in mode and mode != "all_together":
        # randomly selected token to make sure that they don't report as correct
        return torch.IntTensor([0]).to(device), torch.IntTensor([1]).to(device)
    #start_token_id, end_token_id = get_start_and_end_token_ids(category=category, mode=mode)

#    start = 0
#    end = len(ground_truth_tokens) #exclusive
#    start_found = False
#    end_found = False
#    for i in range(len(ground_truth_tokens)):
#        if ground_truth_tokens[i] == start_token_id:
#            start = i
#            start_found = True
#        if ground_truth_tokens[i] == end_token_id:
#            end = i
#            end_found = True
##            if start_found:
##                break

    one_ground_tokens = get_one_span(ground_truth_tokens, category=category) #, start_found, end_found, start, end
    one_predicted_tokens= get_one_span(predicted_tokens, category=category) #, _, _, _, _

#    if mode == "all_together" and not start_found and not end_found:
#        return torch.IntTensor([0]).to(device), torch.IntTensor([1]).to(device)

    return one_ground_tokens, one_predicted_tokens #one_predicted_tokens[-len(one_ground_tokens):] #predicted_tokens[start+1:end] #ground_truth_tokens[start+1:end], predicted_tokens[start+1:end]

def get_one_span(tokens, category="head", ground=True):
    if category not in mode and mode != "all_together":
        # randomly selected token to make sure that they don't report as correct
        return torch.IntTensor([0]).to(device) if ground else torch.IntTensor([1]).to(device)

    start_token_id, end_token_id = get_start_and_end_token_ids(category=category, mode=mode)
    start = 0
    end = len(tokens) #exclusive
    start_found = False
    end_found = False
    for i in range(len(tokens)):
        if tokens[i] == start_token_id:
            start = i
            start_found = True
        if tokens[i] == end_token_id:
            end = i
            end_found = True
            break
    return tokens[start+1:end]#, start_found, end_found, start, end

def validate_head_or_tail(tokens):
    if len(tokens) == 0:
        return False
    set_id_of_special_tokens = set(id_of_special_tokens)
    for token in tokens:
        if token in set_id_of_special_tokens:
            return False
    return True

def validate_reln(tokens):
    if len(tokens) == 0:
        return False
    permitted_reln_tokens = set(tokenizer.convert_tokens_to_ids(relations))
    for token in tokens:
        if token not in permitted_reln_tokens:
            return False
    return True

def validate_predicted_tokens(predicted_tokens):
    for category in mode.split("_"):
        start_token_id, end_token_id = get_start_and_end_token_ids(category=category, mode=mode)
        one_predicted_tokens, _, _, _, _ = get_one_span(predicted_tokens, start_token_id, end_token_id)
        if category == "reln":
            if not validate_reln(one_predicted_tokens):
                return False
        elif category in ["head", "tail"]:
            if not validate_head_or_tail(one_predicted_tokens):
                return False
    return True

def calculate_token_wise_accuracy(b_labels, most_likely_tokens, b_input_ids):
    all_tokens = []

    prediction_per_ground = int(most_likely_tokens.size(0)/b_input_ids.size(0))

    for i in range(b_input_ids.size(0)):

        # process ground_truth_labels

        start_pos = 0
        for j in range(b_input_ids.size(1)):
            if b_input_ids[i,j] == series[0]:
                start_pos = j
            if b_input_ids[i,j] == tokenizer.eos_token_id:
                end_pos = j

        ground_truth_tokens = b_input_ids[i][start_pos:end_pos+1]


        ground_head = get_one_span(ground_truth_tokens, category="head")
        ground_reln = get_one_span(ground_truth_tokens, category="reln")
        ground_tail = get_one_span(ground_truth_tokens, category="tail")

        # process ground truth sentence

        location_of_first_head_token = [k for k in range(len(b_input_ids[i])) if b_input_ids[i][k] == series[0]][0]

        location_of_first_bos_token = [k for k in range(len(b_input_ids[i])) if b_input_ids[i][k] == tokenizer.bos_token_id][0]

        ground_sentence = b_input_ids[i][location_of_first_bos_token+1:location_of_first_head_token]


        # process predicted_labels

        for i_predicted in range(prediction_per_ground):

            start_pos = 0
            for j in range(most_likely_tokens.size(1)):
                if most_likely_tokens[i*prediction_per_ground+i_predicted,j] == series[0]:
                    start_pos = j
                if most_likely_tokens[i*prediction_per_ground+i_predicted,j] == tokenizer.eos_token_id:
                    end_pos = j

            predicted_tokens = most_likely_tokens[i*prediction_per_ground+i_predicted][start_pos:end_pos+1]


            predicted_head = get_one_span(predicted_tokens, category="head", ground=False)
            predicted_reln = get_one_span(predicted_tokens, category="reln", ground=False)
            predicted_tail = get_one_span(predicted_tokens, category="tail", ground=False)

            all_tokens.append([ground_head, predicted_head, ground_reln, predicted_reln, ground_tail, predicted_tail, ground_sentence])

    return all_tokens

def remove_punct(tail):
    return tail.replace('.','').replace('!','').replace('?','').replace(',','')

def correct_one_tail(tail, sentence):
    sentence_split = sentence.split()
    tail_split = tail.split()

    for i in range(len(sentence_split)-len(tail_split)):
        if sentence_split[i:i+len(tail_split)] == tail_split:
            return tail

    if ''.join(tail_split) in sentence_split:
        return remove_punct(''.join(tail_split))

    for i in range(len(sentence_split)-len(tail_split)):
        if ' '.join(sentence_split[i:i+len(tail_split)]).startswith(' '.join(tail_split)):
            return remove_punct(' '.join(sentence_split[i:i+len(tail_split)]))

    if len(tail_split) == 1:
        for j in range(len(tail_split[0]), 1, -1):
            for i in range(len(sentence_split)):
                if sentence_split[i].startswith(tail_split[0][:j]):
                    return remove_punct(sentence_split[i])
    else:
        for i in range(len(sentence_split)):
            if tail_split[0] == sentence_split[i]:
                return remove_punct(' '.join(sentence_split[i:i+len(tail_split)]))

    return tail

def precision_recall_fscore(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1 = (2 * precision * recall) / (precision+recall + 1e-12)
    return precision, recall, f1

def decode_and_save_all_tokens(all_tokens, epoch_equivalent):
    tokens_to_words = {}
    formatted_decoded_tokens = []
    for one_set_of_tokens in all_tokens:
        for one_token in one_set_of_tokens:
            one_decoded_set_of_tokens = []
            for one_small_token in one_token:
                one_token_list = tuple([i for i in one_small_token]) #if 0 <= i < len(tokenizer)
                if one_token_list not in tokens_to_words:
                    try:
                        tokens_to_words[one_token_list] = tokenizer.decode(one_token_list)
                    except TypeError:
                        print("cannot be decoded: ", one_token_list)
                        tokens_to_words[one_token_list] = ''
                one_decoded_set_of_tokens.append(tokens_to_words[one_token_list])
            ground_head, predicted_head, ground_reln, predicted_reln, ground_tail, predicted_tail, ground_sentence= one_decoded_set_of_tokens
            if inference_only and 'original_spans' in generation_name:
                predicted_tail = correct_one_tail(predicted_tail, ground_sentence)

            formatted_decoded_tokens.append({
                        "ground_head":ground_head,
                        "predicted_head":predicted_head,
                        "ground_reln":ground_reln,
                        "predicted_reln":predicted_reln,
                        "ground_tail":ground_tail,
                        "predicted_tail":predicted_tail,
                        "ground_sentence":ground_sentence
                    })

    def process_col(col_name):
        return [i[col_name].strip() for i in formatted_decoded_tokens]

    col_names = ["ground_head","predicted_head",
                 "ground_reln","predicted_reln",
                 "ground_tail", "predicted_tail",
                 "ground_sentence"]

    all_ground_head, all_predicted_head, \
    all_ground_reln, all_predicted_reln, \
    all_ground_tail, all_predicted_tail, \
    all_ground_sentence = [process_col(i) for i in col_names]

#    if inference_only and 'original_spans' in generation_name:
#        # only do this correction for extraction
#        print('doing correction')
#        all_predicted_tail = [correct_one_tail(all_predicted_tail[i], all_ground_sentence[i]) for i in range(len(all_ground_sentence))]
#
    all_ground = [all_ground_head[i] + all_ground_reln[i] + all_ground_tail[i] for i in range(len(all_ground_head))]
    all_predicted = [all_predicted_head[i] + all_predicted_reln[i] + all_predicted_tail[i] for i in range(len(all_ground_head))]

    precision, recall, f1 = precision_recall_fscore(all_ground, all_predicted)

    precision_head, recall_head, f1_head= precision_recall_fscore(all_ground_head, all_predicted_head)
    precision_reln, recall_reln, f1_reln= precision_recall_fscore(all_ground_reln, all_predicted_reln)
    precision_tail, recall_tail, f1_tail = precision_recall_fscore(all_ground_tail, all_predicted_tail)

    df = pd.DataFrame(data=formatted_decoded_tokens)

    if generate_train:
        df.to_csv("{}/train_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))
    elif generate_test:
        df.to_csv("{}/test_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))
    elif generate_custom:
        df.to_csv("{}/custom_tokens_epoch_{}.csv".format(config_name, generate_json_filename.split('/')[-1].replace(".json","")))
    else:
        df.to_csv("{}/eval_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))

    return precision, recall, f1, \
           precision_head, recall_head, f1_head,\
           precision_reln, recall_reln, f1_reln, \
           precision_tail, recall_tail, f1_tail,


def print_example_sentence():

    if mode != "all_together":
        location_of_first_head_token_true = [i for i in range(len(b_input_ids[0])) if b_input_ids[0][i] == series[0]][0]
        location_of_first_head_token_predicted = ([0]+[i for i in range(len(output_undecoded[0])) if output_undecoded[0][i] == series[0]])[-1]
    else:
        location_of_first_head_token_true = [i for i in range(len(b_input_ids[0])) if b_input_ids[0][i] in series[:3]][0]
        location_of_first_head_token_predicted = ([0]+[i for i in range(len(output_undecoded[0])) if output_undecoded[0][i] in series[:3]])[-1]

    print(" true: ", tokenizer.decode(b_input_ids[0][location_of_first_head_token_true:], skip_special_tokens=False))
    if len(output_undecoded[0][location_of_first_head_token_predicted:]):
        decoded_sample = tokenizer.decode(output_undecoded[0][location_of_first_head_token_predicted:], skip_special_tokens=False)
    else:
        decoded_sample = ""
    print(" predicted: ", decoded_sample)


def eval_once(epoch_equivalent):

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0
    total_eval_ppl = 0

    all_tokens = []

    for step, batch in tqdm(enumerate(validation_dataloader), position=0, leave=True):

        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_sample_weights = batch[3]
        b_generate_input_ids = batch[4].to(device)
        b_generate_attn_masks = batch[5].to(device)

        with torch.no_grad():

            outputs  = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask = b_masks,
                             labels=b_labels)

            loss = outputs[0]
            b_logits = outputs[1]
            
            # Shift so that tokens < n predict n
            shift_logits = b_logits[..., :-1, :].contiguous()
            shift_labels = b_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            new_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_per_sample = torch.mean(new_loss.view(shift_labels.size()), dim=-1)

        batch_loss = loss.item()
        total_eval_loss += batch_loss
        total_eval_ppl += 2**batch_loss

        sequences_scores = None
        if need_generation:
            output_undecoded, sequences_scores = generate(b_generate_input_ids, b_generate_attn_masks)

        else:
            output_undecoded = torch.argmax(b_logits, axis=-1)

        print(sequences_scores) 
        raise ValueError

        tokens = calculate_token_wise_accuracy(b_labels, output_undecoded, b_input_ids) #scores,

        all_tokens.append(tokens)

        if generate_one_batch_only:
            break



        if step % sample_every == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, total_epochs))
            print(' Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}'.format(step, len(validation_dataloader), batch_loss, elapsed))
#            print(' Average_correct: {} All_correct: {}'.format(total_eval_average_correct/(step+1),total_eval_all_correct/(step+1)))
#            print(' All_head_correct: {} '.format(total_eval_all_head_correct/(step+1)))
#            print(' All_reln_correct: {} '.format(total_eval_all_reln_correct/(step+1)))
#            print(' All_tail_correct: {} '.format(total_eval_all_tail_correct/(step+1)))
            print(' ppl {} '.format(total_eval_ppl/(step+1)))

            if not inference_only:
                #print_example_sentence()
                pass


    avg_eval_loss = total_eval_loss / len(validation_dataloader)
    avg_eval_ppl = total_eval_ppl / len(validation_dataloader)

    save_tokens_name = generation_name if inference_only else epoch_equivalent

    precision, recall, f1, \
    precision_head, recall_head, f1_head,\
    precision_reln, recall_reln, f1_reln, \
    precision_tail, recall_tail, f1_tail = decode_and_save_all_tokens(all_tokens, save_tokens_name)

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_eval_loss))
    print("  Validation took: {:}".format(validation_time))

    if inference_only:

        if os.path.exists(eval_stats_filename):
            with open(eval_stats_filename) as f:
                eval_stats = [{k: v for k, v in row.items()}
                    for row in csv.DictReader(f, skipinitialspace=True)]
        else:
            eval_stats = []

        current_eval_stats = {
                'Generation Name': generation_name,
                'Valid. Loss': avg_eval_loss,
                'Validation Time': validation_time,
                'avg_eval_ppl':avg_eval_ppl,
                'precision':precision,
                'recall':recall,
                'f1': f1,
                'precision_head':precision_head,
                'recall_head':recall_head,
                'f1_head': f1_head,
                'precision_reln':precision_reln,
                'recall_reln':recall_reln,
                'f1_reln': f1_reln,
                'precision_tail':precision_tail,
                'recall_tail':recall_tail,
                'f1_tail': f1_tail,
            }
        print(current_eval_stats)
        eval_stats.append(current_eval_stats)

        pd.set_option('precision', 5)
        df_stats = pd.DataFrame(data=eval_stats)
        df_stats = df_stats.set_index('Generation Name')
        df_stats.to_csv(eval_stats_filename)

        raise ValueError


    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_equivalent,
            'config_name':config_name,
            'Training Loss': avg_train_loss,
            'avg_train_ppl':avg_train_ppl,
            'Valid. Loss': avg_eval_loss,
            'Training Time': training_time,
            'Validation Time': validation_time,
            'avg_eval_ppl':avg_eval_ppl,
            'precision':precision,
            'recall':recall,
            'f1': f1,
            'precision_head':precision_head,
            'recall_head':recall_head,
            'f1_head': f1_head,
            'precision_reln':precision_reln,
            'recall_reln':recall_reln,
            'f1_reln': f1_reln,
            'precision_tail':precision_tail,
            'recall_tail':recall_tail,
            'f1_tail': f1_tail,
        }
    )


    all_f1s = [i['f1'] for i in training_stats]
    best_f1_position = np.argmax(all_f1s)

    if training_stats[best_f1_position]["f1"] == 0:
        all_f1s = [i['f1_head'] + i['f1_reln'] + i['f1_tail'] for i in training_stats]
        best_f1_position = np.argmax(all_f1s)


    print("best so far by f1 (all)")
    for factor in ['epoch', 'f1', 'precision', 'recall']: #, 'avg_eval_all_correct'
        print("{}: {}".format(factor, training_stats[best_f1_position][factor]))

    pd.set_option('precision', 5)
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')
    df_stats.to_csv(training_stats_filename)
    if not inference_only:
        if int(epoch_equivalent) == round(epoch_equivalent,2):
            save(model, optimizer,scheduler, checkpointer_name, epoch_i)
        if training_stats[best_f1_position]['epoch'] == epoch_equivalent:
            save(model, optimizer,scheduler, best_checkpointer_name, epoch_i)



total_t0 = time.time()

model = model.to(device)

total_epochs = min(max_epochs, epochs+starting_epoch)

for epoch_i in range(starting_epoch, total_epochs):

    if not inference_only:
        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, total_epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0
        total_train_ppl = 0

        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_sample_weights = batch[3]
            b_generate_input_ids = batch[4].to(device)
            b_generate_attn_masks = batch[5].to(device)

            model.zero_grad()


            outputs = model(  b_input_ids,
                              labels=b_labels,
                              attention_mask = b_masks,
                              token_type_ids=None
                            )

            loss = outputs[0]

            if adjust_sample_weight:
                loss.data = loss.data * np.sum(b_sample_weights.detach().numpy())

            b_logits = outputs[1]
            batch_loss = loss.item()
            total_train_loss += batch_loss
            total_train_ppl += 2**batch_loss

            sequences_scores = None
            if need_generation and train_generation:
                output_undecoded, sequences_scores = generate(b_generate_input_ids, b_generate_attn_masks)
            else:
                output_undecoded = torch.argmax(b_logits, axis=-1)

            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs+starting_epoch))
                print(' Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}'.format(step, len(train_dataloader), batch_loss, elapsed))
                print(' ppl {} '.format(total_train_ppl/(step+1)))


                #print_example_sentence()

                model.train()

            loss.backward()

            optimizer.step()

            scheduler.step()


            if (step+1) % int(eval_every * len(train_dataloader)) == 0 and step != 0:
                avg_train_loss = total_train_loss / (step+1)
                avg_train_ppl = total_train_ppl/ (step+1)
                training_time = format_time(time.time() - t0)

                eval_once(epoch_i + (step+1)/len(train_dataloader))

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

    else:
        eval_once(epoch_i + 1)
        break

    #eval_once(epoch_i + 1)

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
