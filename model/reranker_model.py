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
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer,RobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoTokenizer
from reranker_dataset import DiscriminatorDNLIDataset
from torch import nn
import argparse
from sklearn import metrics


parser = argparse.ArgumentParser(description='train_model')
parser.add_argument("--debug_mode", type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument("--load_trained_model", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--lr", default="1e-4") #5e-5
parser.add_argument("--warmup_steps", default="1e2")
parser.add_argument("--config_name", default="default")
parser.add_argument("--inference_only", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Only performs generation without further training; load_trained model must be True")
parser.add_argument("--generation_name", default="",help="top_k-5, top_p-0.5, temperature-0.5, num_beams-5, original_tokens, original_spans")
parser.add_argument("--data_subset", default="all", choices=["all", "within_sentence", "not_within_sentence"])
parser.add_argument("--mode", default="head_reln_tail",choices=[
                                            "head_reln_tail", "head_tail_reln",
                                            "reln_head_tail", "reln_tail_head",
                                            "tail_reln_head", "tail_head_reln",
                                            "head", "reln", "tail"]) # dropped support for all_together
parser.add_argument("--model_name", default="bert", choices=["bert", "roberta"])
parser.add_argument("--generate_train", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--eval_batch_size", type=int, default=10)
parser.add_argument("--train_dataset_filename", default="train_tokens_epoch_all_ten.csv")
parser.add_argument("--generate_test", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--generate_custom", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--random_seed", type=int, default=42)

args = parser.parse_args()

debug_mode = args.debug_mode
load_trained_model = args.load_trained_model
epochs = args.epochs
max_epochs = args.max_epochs
learning_rate = float(args.lr)
warmup_steps = float(args.warmup_steps)
config_name = args.config_name
inference_only = args.inference_only
generation_name = args.generation_name
data_subset = args.data_subset
mode = args.mode #"all_together" #
model_name = args.model_name
generate_train = args.generate_train
batch_size = args.batch_size
train_dataset_filename = args.train_dataset_filename
generate_test = args.generate_test
generate_custom = args.generate_custom
eval_batch_size = min(args.eval_batch_size, batch_size)


sample_every = 300 if not debug_mode else 1

# Set the seed value all over the place to make this reproducible.
seed_val = args.random_seed

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

if not os.path.exists(config_name+"/"):
    os.makedirs(config_name+"/")

checkpointer_name = "{}/discriminator_pytorch_model.pth".format(config_name)
best_checkpointer_name = "{}/discriminator_pytorch_model_best.pth".format(config_name)
training_stats_filename = "{}/discriminator_training_stats.csv".format(config_name)
eval_stats_filename = "{}/discriminator_eval_stats.csv".format(config_name)




eval_every = 0.25

epsilon = 1e-8

adjust_sample_weight = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if model_name == "bert":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
elif model_name == "roberta":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")


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


tokenizer.add_special_tokens({'pad_token': '[PAD]',
                              'bos_token': '[CLS]',
                              'eos_token': '[PAD]',
                              "additional_special_tokens":["[HEAD]", "[RELN]", "[TAIL]"] + relations
                              })

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
model.resize_token_embeddings(len(tokenizer))

optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

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

if not inference_only:
    train_dataset = DiscriminatorDNLIDataset("{}/{}".format(config_name, train_dataset_filename) ,
                                tokenizer, debug_mode=debug_mode, data_subset=data_subset, mode=mode)
else:
    train_dataset = DiscriminatorDNLIDataset("{}/{}".format(config_name, train_dataset_filename) ,
                                tokenizer, debug_mode=True, data_subset=data_subset, mode=mode)

if not generate_custom and "train" not in train_dataset_filename:
    raise ValueError("please name train and test dataset filenames with the only difference of 'train' and 'eval' ")


elif generate_test:
    val_dataset_filename = train_dataset_filename.replace("train", "test")
else:
    val_dataset_filename = train_dataset_filename.replace("train", "eval")

val_dataset = DiscriminatorDNLIDataset("{}/{}".format(config_name, val_dataset_filename),
                          tokenizer, debug_mode=debug_mode, data_subset=data_subset, mode=mode)

#if generate_train:
#    val_dataset = DiscriminatorDNLIDataset("dnli/dialogue_nli/dialogue_nli_train.jsonl",
#                          tokenizer, debug_mode=debug_mode, data_subset=data_subset, mode=mode)
#else:
#    val_dataset = DiscriminatorDNLIDataset("dnli/dialogue_nli/dialogue_nli_dev.jsonl",
#                          tokenizer, debug_mode=debug_mode, data_subset=data_subset, mode=mode)

print('{:>5,} training samples'.format(len(train_dataset)))
print('{:>5,} validation samples'.format(len(val_dataset)))


train_dataloader = DataLoader(
            train_dataset,
            sampler = SequentialSampler(train_dataset), #RandomSampler(train_dataset),
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
                                                       ['[CLS]', '[PAD]'] +
                                                       relations)

head_token_id = tokenizer.convert_tokens_to_ids("[HEAD]")
reln_token_id = tokenizer.convert_tokens_to_ids("[RELN]")
tail_token_id = tokenizer.convert_tokens_to_ids("[TAIL]")
eos_token_id = tokenizer.eos_token_id


if mode != "all_together":
    series = tokenizer.convert_tokens_to_ids(["[{}]".format(i) for i in mode.upper().split("_")]) + [eos_token_id]
else:
    series = [head_token_id, reln_token_id, tail_token_id, eos_token_id]

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

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

def get_one_span(tokens, category="head"):
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

def get_head_reln_tail_tokens(tokens):
    start_pos = 0
    end_pos = len(tokens)-1
    for j in range(tokens.size(-1)):
        if tokens[j] == series[0]:
            start_pos = j
        if tokens[j] == tokenizer.eos_token_id:
            end_pos = j

    relevant_tokens = tokens[start_pos:end_pos+1]

    head_tokens = get_one_span(relevant_tokens, category="head")
    reln_tokens = get_one_span(relevant_tokens, category="reln")
    tail_tokens = get_one_span(relevant_tokens, category="tail")

    return [head_tokens, reln_tokens, tail_tokens]

def calculate_token_wise_accuracy(b_logits, b_input_ids, b_ground_truth_tokens):

    # b_labels --> tensor (batch_size,) # needs to have one correct one

    # b_logits --> tensor (batch_size, n_labels)
    # b_input_ids --> tensor (batch_size, seq_len) --> just take random one)

    # process ground truth sentence

    if mode != "all_together":

        location_of_first_head_token = [k for k in range(len(b_input_ids[0])) if b_input_ids[0][k] == series[0]][0]
#        else:
#            #print(b_input_ids[i])
#            location_of_first_head_token = [k for k in range(len(b_input_ids[i])) if b_input_ids[i][k] in series[:3]][0]
#
    location_of_first_bos_token = [k for k in range(len(b_input_ids[0])) if b_input_ids[0][k] == tokenizer.bos_token_id][0]

    ground_sentence = b_input_ids[0][location_of_first_bos_token+1:location_of_first_head_token]

    ground_truth_tokens = b_ground_truth_tokens[0]

    highest_prob_index = torch.argmax(b_logits[:, 1])

    confidence = torch.nn.Softmax(dim=0)(b_logits[:, 1])[highest_prob_index].item()

    confidence = b_logits[highest_prob_index, 1].item()

    predicted_tokens = b_input_ids[highest_prob_index]

    ground_head, ground_reln, ground_tail = get_head_reln_tail_tokens(ground_truth_tokens)
    predicted_head, predicted_reln, predicted_tail = get_head_reln_tail_tokens(predicted_tokens)

    all_tokens = [[ground_head, predicted_head, ground_reln, predicted_reln, ground_tail, predicted_tail, ground_sentence, confidence]]

    return all_tokens


def precision_recall_fscore(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1 = (2 * precision * recall) / (precision+recall+ 1e-12)
    return precision, recall, f1

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

def decode_and_save_all_tokens(all_tokens, epoch_equivalent):
    tokens_to_words = {}
    formatted_decoded_tokens = []
    for one_set_of_tokens in all_tokens:
        for one_token in one_set_of_tokens:
            one_decoded_set_of_tokens = []
            for one_small_token in one_token[:-1]:
                one_token_list = tuple([i for i in one_small_token]) #if 0 <= i < len(tokenizer)
                if one_token_list not in tokens_to_words:
                    try:
                        tokens_to_words[one_token_list] = tokenizer.decode(one_token_list)
                    except TypeError:
                        print("cannot be decoded: ", one_token_list)
                        tokens_to_words[one_token_list] = ''
                one_decoded_set_of_tokens.append(tokens_to_words[one_token_list])
            #this is confidence
            one_decoded_set_of_tokens.append(one_token[-1])

            ground_head, predicted_head, ground_reln, predicted_reln, ground_tail, predicted_tail, ground_sentence, confidence = one_decoded_set_of_tokens
            formatted_decoded_tokens.append({
                        "ground_head":ground_head,
                        "predicted_head":predicted_head,
                        "ground_reln":ground_reln,
                        "predicted_reln":predicted_reln,
                        "ground_tail":ground_tail,
                        "predicted_tail":predicted_tail,
                        "ground_sentence":ground_sentence,
                        "confidence":confidence
                    })

    all_ground_head = [i["ground_head"] for i in formatted_decoded_tokens]
    all_predicted_head = [i["predicted_head"] for i in formatted_decoded_tokens]
    all_ground_reln = [i["ground_reln"] for i in formatted_decoded_tokens]
    all_predicted_reln = [i["predicted_reln"] for i in formatted_decoded_tokens]
    all_ground_tail = [i["ground_tail"] for i in formatted_decoded_tokens]
    all_predicted_tail = [i["predicted_tail"] for i in formatted_decoded_tokens]
    all_ground_sentence = [i["ground_sentence"] for i in formatted_decoded_tokens]

    if 'not' not in config_name:
        # only do this correction for extraction
        all_predicted_tail = [correct_one_tail(all_predicted_tail[i], all_ground_sentence[i]) for i in range(len(all_ground_sentence))]

    all_ground = [all_ground_head[i] + all_ground_reln[i] + all_ground_tail[i] for i in range(len(all_ground_head))]
    all_predicted = [all_predicted_head[i] + all_predicted_reln[i] + all_predicted_tail[i] for i in range(len(all_ground_head))]

    precision, recall, f1 = precision_recall_fscore(all_ground, all_predicted)

    precision_head, recall_head, f1_head = precision_recall_fscore(all_ground_head, all_predicted_head)
    precision_reln, recall_reln, f1_reln = precision_recall_fscore(all_ground_reln, all_predicted_reln)
    precision_tail, recall_tail, f1_tail = precision_recall_fscore(all_ground_tail, all_predicted_tail)

    df = pd.DataFrame(data=formatted_decoded_tokens)

    if generate_train:
        df.to_csv("{}/discriminator_train_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))
    elif generate_custom:
        df.to_csv("{}/discriminator_custom_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))
    elif generate_test:
        df.to_csv("{}/discriminator_test_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))
    else:
        df.to_csv("{}/discriminator_eval_tokens_epoch_{}.csv".format(config_name, epoch_equivalent))

    return precision, recall, f1, \
           precision_head, recall_head, f1_head,\
           precision_reln, recall_reln, f1_reln, \
           precision_tail, recall_tail, f1_tail,

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

#    total_eval_all_correct = 0
#    total_eval_average_correct = 0
#    total_eval_all_head_correct = 0
#    total_eval_average_head_correct = 0
#    total_eval_all_reln_correct = 0
#    total_eval_average_reln_correct = 0
#    total_eval_all_tail_correct = 0
#    total_eval_average_tail_correct = 0

    all_tokens = []

    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(validation_dataloader), position=0, leave=True):

        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_ground_truth_tokens = batch[3].to(device)
#        b_sample_weights = batch[3]
#        b_generate_input_ids = batch[4].to(device)
#        b_generate_attn_masks = batch[5].to(device)

        with torch.no_grad():

            outputs = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_masks,
                             labels=b_labels)

            loss = outputs[0]
            b_logits = outputs[1]


        batch_loss = loss.item()
        total_eval_loss += batch_loss
        total_eval_ppl += 2**batch_loss

#        if need_generation:
#            output_undecoded = generate(b_generate_input_ids, b_generate_attn_masks)
#            #print(" sentence: ", tokenizer.decode(output_undecoded))
#            # save the sentences themselves and use the bert model separately
#            #raise ValueError
#
##            generated_tokens = get_head_reln_tail_from_tokens(output_undecoded)
##            value_embeddings = get_embeddings_from_tokens(generated_tokens)
##            print(torch.stack(value_embeddings).size())
##            ds_embeddings = get_ds_embeddings()
##            print(torch.stack(ds_embeddings).size())
##            raise ValueError
#
#        else:
#            output_undecoded = torch.argmax(b_logits, axis=-1)

        if b_logits.size(0) == eval_batch_size:
            tokens = calculate_token_wise_accuracy(b_logits, b_input_ids, b_ground_truth_tokens) #scores,
            all_tokens.append(tokens)
        else:
            for i in range(0, b_logits.size(0), eval_batch_size):
                tokens = calculate_token_wise_accuracy(b_logits[i:i+eval_batch_size, :], b_input_ids[i:i+eval_batch_size, :], b_ground_truth_tokens[i:i+eval_batch_size, :]) #scores,
                all_tokens.append(tokens)

         # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, total_epochs))
            print(' Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}'.format(step, len(validation_dataloader), batch_loss, elapsed))
#            print(' Average_correct: {} All_correct: {}'.format(total_eval_average_correct/(step+1),total_eval_all_correct/(step+1)))
#            print(' All_head_correct: {} '.format(total_eval_all_head_correct/(step+1)))
#            print(' All_reln_correct: {} '.format(total_eval_all_reln_correct/(step+1)))
#            print(' All_tail_correct: {} '.format(total_eval_all_tail_correct/(step+1)))
            print(' ppl {} '.format(total_eval_ppl/(step+1)))



    avg_eval_loss = total_eval_loss / len(validation_dataloader)
    avg_eval_ppl = total_eval_ppl / len(validation_dataloader)

#    avg_eval_all_correct = total_eval_all_correct / len(validation_dataloader)
#    #avg_eval_average_correct = total_eval_average_correct / len(validation_dataloader)
#
#    avg_eval_all_head_correct = total_eval_all_head_correct / len(validation_dataloader)
#    #avg_eval_average_head_correct = total_eval_average_head_correct / len(validation_dataloader)
#    avg_eval_all_reln_correct = total_eval_all_reln_correct / len(validation_dataloader)
#    #avg_eval_average_reln_correct = total_eval_average_reln_correct / len(validation_dataloader)
#    avg_eval_all_tail_correct = total_eval_all_tail_correct / len(validation_dataloader)
#    #avg_eval_average_tail_correct = total_eval_average_tail_correct / len(validation_dataloader)
#
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

        eval_stats.append({
                'Generation Name': generation_name,
                'Valid. Loss': avg_eval_loss,
                'Validation Time': validation_time,
                'avg_eval_ppl':avg_eval_ppl,
#                'avg_eval_all_correct':avg_eval_all_correct,
#                'avg_eval_average_correct': avg_eval_average_correct,
                'precision':precision,
                'recall':recall,
                'f1': f1,
#                'avg_eval_all_head_correct':avg_eval_all_head_correct,
#                'avg_eval_average_head_correct':avg_eval_average_head_correct,
                'precision_head':precision_head,
                'recall_head':recall_head,
                'f1_head': f1_head,
#                'avg_eval_all_reln_correct':avg_eval_all_reln_correct,
#                'avg_eval_average_reln_correct':avg_eval_average_reln_correct,
                'precision_reln':precision_reln,
                'recall_reln':recall_reln,
                'f1_reln': f1_reln,
#                'avg_eval_all_tail_correct':avg_eval_all_tail_correct,
#                'avg_eval_average_tail_correct':avg_eval_average_tail_correct,
                'precision_tail':precision_tail,
                'recall_tail':recall_tail,
                'f1_tail': f1_tail,
            })
        print(eval_stats)
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
#            'avg_train_all_correct':avg_train_all_correct,
#            'avg_train_average_correct': avg_train_average_correct,
#            'avg_train_all_head_correct':avg_train_all_head_correct,
#            'avg_train_average_head_correct':avg_train_average_head_correct,
#            'avg_train_all_reln_correct':avg_train_all_reln_correct,
#            'avg_train_average_reln_correct':avg_train_average_reln_correct,
#            'avg_train_all_tail_correct':avg_train_all_tail_correct,
#            'avg_train_average_tail_correct':avg_train_average_tail_correct,
            'Valid. Loss': avg_eval_loss,
            'Training Time': training_time,
            'Validation Time': validation_time,
            'avg_eval_ppl':avg_eval_ppl,
#            'avg_eval_all_correct':avg_eval_all_correct,
#            'avg_eval_average_correct': avg_eval_average_correct,
#            'avg_eval_all_head_correct':avg_eval_all_head_correct,
#            'avg_eval_average_head_correct':avg_eval_average_head_correct,
#            'avg_eval_all_reln_correct':avg_eval_all_reln_correct,
#            'avg_eval_average_reln_correct':avg_eval_average_reln_correct,
#            'avg_eval_all_tail_correct':avg_eval_all_tail_correct,
#            'avg_eval_average_tail_correct':avg_eval_average_tail_correct,
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

#        total_train_all_correct = 0
#        total_train_average_correct = 0
#        total_train_all_head_correct = 0
#        total_train_average_head_correct = 0
#        total_train_all_reln_correct = 0
#        total_train_average_reln_correct = 0
#        total_train_all_tail_correct = 0
#        total_train_average_tail_correct = 0

        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_ground_truth_tokens = batch[3].to(device)

#            print(b_input_ids.size())
#            print(b_masks.size())
#            print(b_labels.size())
#            print(b_ground_truth_tokens.size())
#            print(b_labels)
#            print(b_input_ids[0])
#            print(tokenizer.decode(b_input_ids[0]))
#            raise ValueError
            model.zero_grad()


            outputs = model(  b_input_ids,
                              labels=b_labels,
                              attention_mask=b_masks,
                              token_type_ids=None
                            )

            loss = outputs[0]
            b_logits = outputs[1]

            batch_loss = loss.item()
            total_train_loss += batch_loss

            #tokens = calculate_token_wise_accuracy(b_labels, b_logits) #scores,

#
#            average_correct, all_correct, \
#            average_head_correct, all_head_correct, \
#            average_reln_correct, all_reln_correct, \
#            average_tail_correct, all_tail_correct = scores

            total_train_ppl += 2**batch_loss

#            total_train_all_correct += all_correct
#            total_train_average_correct +=average_correct
#            total_train_all_head_correct += all_head_correct
#            total_train_average_head_correct += average_head_correct
#            total_train_all_reln_correct += all_reln_correct
#            total_train_average_reln_correct += average_reln_correct
#            total_train_all_tail_correct += all_tail_correct
#            total_train_average_tail_correct += average_tail_correct


            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs+starting_epoch))
                print(' Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}'.format(step, len(train_dataloader), batch_loss, elapsed))
#                print(' Average_correct: {} All_correct: {}'.format(total_train_average_correct/(step+1),total_train_all_correct/(step+1)))
#                print(' Average_head_correct: {} All_head_correct: {} '.format(total_train_average_head_correct/(step+1),total_train_all_head_correct/(step+1)))
#                print(' Average_reln_correct: {} All_reln_correct: {} '.format(total_train_average_reln_correct/(step+1),total_train_all_reln_correct/(step+1)))
#                print(' Average_tail_correct: {} All_tail_correct: {} '.format(total_train_average_tail_correct/(step+1),total_train_all_tail_correct/(step+1)))
                print(' ppl {} '.format(total_train_ppl/(step+1)))


                model.train()

            loss.backward()

            optimizer.step()

            scheduler.step()

            if (step+1) % int(eval_every * len(train_dataloader)) == 0 and step != 0:
                # Calculate the average loss over all of the batches.
                avg_train_loss = total_train_loss / (step+1)
                avg_train_ppl = total_train_ppl/ (step+1)
#                avg_train_all_correct = total_train_all_correct / (step+1)
#                avg_train_average_correct = total_train_average_correct / (step+1)
#                avg_train_all_head_correct = total_train_all_head_correct / (step+1)
#                avg_train_average_head_correct = total_train_average_head_correct / (step+1)
#                avg_train_all_reln_correct = total_train_all_reln_correct / (step+1)
#                avg_train_average_reln_correct = total_train_average_reln_correct / (step+1)
#                avg_train_all_tail_correct = total_train_all_tail_correct / (step+1)
#                avg_train_average_tail_correct = total_train_average_tail_correct / (step+1)
                # Measure how long this epoch took.
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
