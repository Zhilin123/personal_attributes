#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
from tqdm import trange
import torch
import json
from collections import defaultdict
import numpy as np
import pandas as pd

class DiscriminatorDNLIDataset(Dataset):
    
    def __init__(self, filename,  tokenizer, max_length=128, debug_mode=False, data_subset="all", mode="head_reln_tail"):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # in the format [CLS] sentence [HEAD] head_tokens [RELN] reln_tokens [TAIL] tail_tokens [PAD] [PAD]
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        # in the format [CLS] sentence [HEAD] head_tokens [RELN] reln_tokens [TAIL] tail_tokens [PAD] [PAD]
        self.ground_truth_tokens = []
        
        df = pd.read_csv(filename)
        
        ground_sentence = [str(i) for i in list(df["ground_sentence"])]
        
        ground_head = [str(i) for i in list(df["ground_head"])]
        ground_reln = [str(i) for i in list(df["ground_reln"])]
        ground_tail = [str(i) for i in list(df["ground_tail"])]
        
        predicted_head = [str(i) for i in list(df["predicted_head"])]
        predicted_reln = [str(i) for i in list(df["predicted_reln"])]
        predicted_tail = [str(i) for i in list(df["predicted_tail"])]
        
        if debug_mode:
            ground_sentence = ground_sentence[:60]
        
        element_to_tokenized = {}
        
        for i in trange(len(ground_sentence)):
            
            ground_elements = ["[CLS]", ground_sentence[i], 
                                 "[HEAD]", ground_head[i], 
                                 "[RELN]", ground_reln[i], 
                                 "[TAIL]", ground_tail[i], 
                                 "[PAD]"]
            
            if tuple(ground_elements) not in element_to_tokenized:
                ground_dict = self.tokenizer(" ".join(ground_elements), 
                                                truncation=True,
                                                max_length=self.max_length, 
                                                padding="max_length",
                                                return_tensors="pt")
                
                element_to_tokenized[tuple(ground_elements)] = ground_dict
            else:
                ground_dict = element_to_tokenized[tuple(ground_elements)]
            
            ground_input_ids = torch.squeeze(ground_dict['input_ids'])
            #ground_attn_masks = torch.squeeze(ground_dict['attention_mask'])
            
            predicted_elements = ["[CLS]", ground_sentence[i], 
                                     "[HEAD]", predicted_head[i], 
                                     "[RELN]", predicted_reln[i], 
                                     "[TAIL]", predicted_tail[i], 
                                     "[PAD]"]
            
            if tuple(predicted_elements) not in element_to_tokenized:
                predicted_dict = self.tokenizer(" ".join(predicted_elements), 
                                                truncation=True,
                                                max_length=self.max_length, 
                                                padding="max_length",
                                                return_tensors="pt",
                                                add_special_tokens=False)
                element_to_tokenized[tuple(predicted_elements)] = predicted_dict
            else:
                predicted_dict = element_to_tokenized[tuple(predicted_elements)]
                
            
            predicted_input_ids = torch.squeeze(predicted_dict['input_ids'])
            predicted_attn_masks = torch.squeeze(predicted_dict['attention_mask'])
            
            label = int((predicted_head[i] + predicted_reln[i] + predicted_tail[i]) == \
                        (ground_head[i] + ground_reln[i] + ground_tail[i]))
            
            self.labels.append(label)
            self.ground_truth_tokens.append(ground_input_ids)
            self.input_ids.append(predicted_input_ids)
            self.attn_masks.append(predicted_attn_masks)
    
        del element_to_tokenized
        
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, item):
        return self.input_ids[item], self.attn_masks[item], self.labels[item], self.ground_truth_tokens[item]
                