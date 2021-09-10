#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from tqdm import trange
import torch
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import os

def save_jsonl(docs, output_filename):
    with open(output_filename,'w+') as f:
        for relation in docs:
            f.write(json.dumps(relation))
            f.write('\n')

def save_json(docs, output_filename):
    with open(output_filename, "w") as write_file:
        json.dump(docs, write_file)

def save_txt(docs, output_filename):
    f = open(output_filename, "w+")
    for item in docs:
        f.write(item)
        f.write('\n')
    f.close()

def get_start_end_pos(text, triple, system="dygiepp"):
    text_split = text.split()
    head_split = triple[0].split()
    tail_split = triple[2].split()

    head_pos = None
    tail_pos = None
    for i in range(len(text_split)-len(head_split)):
        if text_split[i:i+len(head_split)] == head_split and head_pos is None:
            head_pos = (i, i+len(head_split))
        elif text_split[i:i+len(tail_split)] == tail_split and tail_pos is None:
            tail_pos = (i, i+len(tail_split))

    if head_pos and tail_pos:
        if system in ["dygiepp", "nyt", "nyt_inference"]:
            return head_pos[0], head_pos[1]-1, tail_pos[0], tail_pos[1]-1
        else:
            return head_pos[0], head_pos[1], tail_pos[0], tail_pos[1]
    else:
        return None, None, None, None

def process_sentence_into_ace_format(new_sentences, new_triples, output_filename, system="dygiepp"):

    folder = "/".join(output_filename.split("/")[:-1])
    if not os.path.exists(folder+"/"):
        os.makedirs(folder+"/")
            
    docs = []

    for j in trange(len(new_sentences)):
        text = new_sentences[j]
        triple = new_triples[j]
        sentences = text.split('\n')
        if len(sentences) > 1:
            raise ValueError
        sentences = [line.split() for line in sentences]
        #sentcount += len(sentences)
        sentence_ids = []
        i = 0
        for sentence in sentences:
            ids = []
            for word in sentence:
                ids.append(i)
                i += 1
            sentence_ids.append(ids)
        head_start, head_end, tail_start, tail_end = get_start_end_pos(text, triple, system=system)
        ner = [[] for i in range(len(sentences))]
        relations = [[] for i in range(len(sentences))]
        relations[0].append([head_start, head_end, tail_start, tail_end, new_triples[j][1]])
        if head_start is not None:
            docs.append({"sentences":sentences, "ner":ner, "relations": relations, "clusters":[], "doc_key":str(j)})

    print(len(docs))
    save_jsonl(docs, output_filename)


def process_sentence_into_nyt_format(new_sentences, new_triples, output_filename_wo_format, system="nyt"):

    """
    # we need to output
    .sent: mr. scruggs -- who is arguing the case with his son zach ; a colleague from northern mississippi , don barrett ; and john jones from jackson , miss. -- said he had already filed suit on behalf of about 2,000 clients seeking redress from their insurers , including senator trent lott , who is both a neighbor of mr. scruggs in pascagoula and his brother-in-law ; and representative gene taylor of bay st. louis .
    .tup: trent lott ; mississippi ; /people/person/place_lived
    .pointer: 50 51 17 17 /people/person/place_lived
    .dep: a jsonl format {"adj_mat": [[0 * len(sent)]*len(sent)]}
    """

    folder = "/".join(output_filename_wo_format.split("/")[:-1])
    if not os.path.exists(folder+"/"):
        os.makedirs(folder+"/")
        
    def convert_tup(triple):
        return ' ; '.join([triple[0], triple[2], triple[1]])

    sent = []

    tup = []

    pointer = []

    dep = []

    for j in trange(len(new_sentences)):
        text = new_sentences[j]
        triple = new_triples[j]
        head_start, head_end, tail_start, tail_end = get_start_end_pos(text, triple, system=system)

        if head_start is not None or system == "nyt_inference":
            sent.append(text)
            tup.append(convert_tup(triple))
            pointer.append(' '.join([str(head_start), str(head_end), str(tail_start), str(tail_end), triple[1]]))
            len_text = len(text.split())
            one_dep = [[0] * len_text] *len_text
            dep.append({
                        "adj_mat":one_dep
                    })


    save_jsonl(dep, output_filename_wo_format + ".dep")
    save_txt(tup, output_filename_wo_format + ".tup")
    save_txt(sent, output_filename_wo_format + ".sent")
    save_txt(pointer, output_filename_wo_format + ".pointer")
    save_txt(list(set([triple[1]for triple in new_triples])),output_filename_wo_format+".relation")

class DNLIDataset(Dataset):

    def __init__(self, filename,  tokenizer, max_length=128, debug_mode=False, data_subset="all", mode="head_reln_tail", unified_ontology=False):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.lm_labels = []
        self.sample_weights = []
        self.generate_input_ids = []
        self.generate_attn_masks = []
        self.max_length = max_length
        self.mode = mode

        if "dnli/dialogue_nli" in filename:
            all_sentences, all_triples = DNLIDataset.load_sentences_and_triples(filename)
            all_sentences, all_triples = DNLIDataset.remove_duplicates(all_sentences, all_triples)
            if unified_ontology:
                all_sentences, all_triples = DNLIDataset.remove_invalid_triples(all_sentences, all_triples)
                all_triples = DNLIDataset.remove_tail_prefix_numbers(all_triples)
                all_triples = DNLIDataset.unify_ontology(all_triples)
            
            if data_subset == "within_sentence":
                all_sentences, all_triples, _, _= DNLIDataset.retain_triples_found_in_sentences_only(all_sentences, all_triples)
            elif data_subset == "not_within_sentence":
                _, _, all_sentences, all_triples = DNLIDataset.retain_triples_found_in_sentences_only(all_sentences, all_triples)
        else:
            all_sentences = json.load(open(filename, "r"))
            all_triples = [('i', 'physical_attribute', 'blonde')] * len(all_sentences)

        self.triples = all_triples
        self.sentences = all_sentences
        self.set_of_relations = set()

        self.head_weight = self.calculate_loss_weight(mode="head")
        self.reln_weight = self.calculate_loss_weight(mode="reln")

        if debug_mode:
            all_sentences = all_sentences[:64]
            all_triples = all_triples[:64]

        for i in trange(len(all_sentences)):

            txt = all_sentences[i]
            triple = all_triples[i]

            self.set_of_relations.add("["+triple[1]+"]")

            if mode != "all_together":
                input_ids, labels, attn_masks, sample_weight, \
                generate_input_ids, generate_attn_masks = self.get_formated_input(txt, triple, mode=self.mode)

                self.input_ids.append(input_ids)
                self.lm_labels.append(labels)
                self.attn_masks.append(attn_masks)
                self.sample_weights.append(sample_weight)
                self.generate_input_ids.append(generate_input_ids)
                self.generate_attn_masks.append(generate_attn_masks)
            else:
                for small_mode in ["head", "reln", "tail"]:
                    input_ids, labels, attn_masks, sample_weight, \
                    generate_input_ids, generate_attn_masks = self.get_formated_input(txt, triple, mode=small_mode)


                    self.input_ids.append(input_ids)
                    self.lm_labels.append(labels)
                    self.attn_masks.append(attn_masks)
                    self.sample_weights.append(sample_weight)
                    self.generate_input_ids.append(generate_input_ids)
                    self.generate_attn_masks.append(generate_attn_masks)


    def get_formated_input(self, txt, triple, mode=None):

        first_special_token = "[{}]".format(mode.upper().split("_")[0])

        special_token_to_value = {
                    "[HEAD]":triple[0],
                    "[RELN]":"["+triple[1]+"]",
                    "[TAIL]":triple[2]
                }

        list_for_text1 = []

        for i in mode.upper().split("_"):
            special_token = "[{}]".format(i)
            list_for_text1.append(special_token)
            list_for_text1.append(special_token_to_value[special_token])

        txt1 = ' '.join(list_for_text1)

        to_encode = '<|startoftext|>' + txt + ' ' + txt1 + '<|endoftext|>'


        encodings_dict = self.tokenizer(to_encode,
                                        truncation=True,
                                        max_length=self.max_length,
                                        padding="max_length")

        pre_tensor_input_ids = encodings_dict['input_ids']


        first_special_token_id = self.tokenizer.convert_tokens_to_ids([first_special_token])[0]
        position_of_first_token = [j for j in range(len(encodings_dict['input_ids'])) if encodings_dict['input_ids'][j] == first_special_token_id][0]
        position_of_start_token = [j for j in range(len(encodings_dict['input_ids'])) if encodings_dict['input_ids'][j] == self.tokenizer.bos_token_id][0]
        pre_tensor_labels = [-100 if position_of_start_token <= i < position_of_first_token+1 else encodings_dict['input_ids'][i] for i in range(len(encodings_dict['input_ids']))]

        generate_encodings_dict = self.tokenizer('<|startoftext|>' + txt + " " +
                                                 first_special_token, max_length=self.max_length-32,
                                                 padding="max_length", return_tensors="pt")

        generate_input_ids=torch.squeeze(generate_encodings_dict['input_ids'])
        generate_attn_masks=torch.squeeze(generate_encodings_dict['attention_mask'])
        input_ids = torch.tensor(pre_tensor_input_ids)
        attn_masks = torch.tensor(encodings_dict['attention_mask'])
        labels = torch.tensor(pre_tensor_labels)
        sample_weight = (self.head_weight[triple[0]]) + self.reln_weight[triple[1]]

        return input_ids, labels, attn_masks, sample_weight, generate_input_ids, generate_attn_masks


    def calculate_loss_weight(self, mode="head"):
        if mode == "head":
            samples = [i[0] for i in self.triples]
        elif mode == "reln":
            samples = [i[1] for i in self.triples]

        counter = defaultdict(int)
        for sample in samples:
            counter[sample] += 1
        weight = defaultdict(float)
        for sample in counter:
            weight[sample] = 1 / counter[sample]
        return weight

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.attn_masks[item], self.lm_labels[item],\
               self.sample_weights[item], self.generate_input_ids[item], self.generate_attn_masks[item]


    @staticmethod
    def load_sentences_and_triples(filename):
        with open(filename, 'r') as json_file:
            json_list = json_file

            for json_str in json_list:
                result = json.loads(json_str)

        all_sentences = []
        all_triples = []

        for one_result in result:
            for i in range(1,3):
                all_sentences.append(one_result[f"sentence{i}"])
                all_triples.append(one_result[f"triple{i}"])
        return all_sentences, all_triples

    @staticmethod
    def remove_duplicates(sentences, triples):
        sentence_triple_pairs = {}
        for i in range(len(sentences)):
            if "<none>" not in triples[i] and "<blank>" not in triples[i]:
                one_pair = tuple([sentences[i],tuple(triples[i])])
                sentence_triple_pairs[one_pair] = 1
        all_keys = list(sentence_triple_pairs.keys())
        all_sentences = [i[0] for i in all_keys]
        all_triples = [i[1] for i in all_keys]
        return all_sentences, all_triples

    @staticmethod
    def retain_triples_found_in_sentences_only(all_sentences, all_triples):
        new_sentences = []
        new_triples = []
        rejected_sentences = []
        rejected_triples = []
        for i in range(len(all_sentences)):
            sentence = all_sentences[i]
            triple = all_triples[i]
            head = triple[0]
            tail = triple[2]

            text_split = sentence.split()
            head_split = head.split()
            tail_split = tail.split()
            contains_head = False
            contains_tail = False


            for j in range(len(text_split)-len(head_split)):
                if text_split[j:j+len(head_split)] == head_split:
                    contains_head = True

            for j in range(len(text_split)-len(tail_split)):
                if text_split[j:j+len(tail_split)] == tail_split:
                    contains_tail = True

            if contains_tail and contains_head:
                new_sentences.append(sentence)
                new_triples.append(triple)
            else:
                rejected_sentences.append(sentence)
                rejected_triples.append(triple)

        return new_sentences, new_triples, rejected_sentences, rejected_triples

    @staticmethod
    def remove_invalid_triples(sentences, triples):
        invalid_relns = set(["have", "not_have", "other", "like_general", "want", "dislike", "favorite"])
        invalid_tail_fragments = ["<blank>", "unspecified", "unknown"]
        clean_sentences = []
        clean_triples = []
        for i in range(len(sentences)):
            triple = triples[i]
            head, reln, tail = triple
            if reln in invalid_relns:
                continue
            any_fragment = sum([int(fragment in tail) for fragment in invalid_tail_fragments])
            if any_fragment:
                continue
            clean_sentences.append(sentences[i])
            clean_triples.append(triples[i])
        return clean_sentences, clean_triples
    
    @staticmethod
    def remove_tail_prefix_numbers(triples):
        clean_triples = []
        for i in range(len(triples)):
            triple = triples[i]
            head, reln, tail = triple
            tail_split = tail.split()
            if len(tail_split) > 1 and tail_split[0].isdigit():
                tail = " ".join(tail_split[1:])
            clean_triples.append((head, reln, tail))
        return clean_triples
    
    @staticmethod
    def unify_ontology(triples):
        src_to_tgt = {
                    "favorite_food": "like_food",
                    "favorite_drink": "like_drink",
                    "favorite_show": "like_watching",
                    "favorite_sport": "like_sports",
                    "favorite_book": "like_read",
                    "favorite_place": "like_goto",
                    "favorite_movie": "like_movie",
                    "favorite_music": "like_music",
                    "favorite_animal": "like_animal",
                    "has_hobby": "like_activity",
                    "favorite_hobby": "like_activity",
                    "favorite_activity": "like_activity",
                    "have_chidren": "have_family",
                    "have_sibling": "have_family"
                }
        
        unified_triples = []
        
        for triple in triples:
            head, reln, tail = triple
            if reln in src_to_tgt:
                reln = src_to_tgt[reln]
            unified_triples.append((head, reln, tail))
            
        return unified_triples


def get_descriptive_stats(all_sentences, all_triples):
    words_in_sentence = [len(i.split()) for i in all_sentences]
    words_in_head_entity = [len(j.split()) for j in [i[0] for i in all_triples]]
    words_in_tail_entity = [len(j.split()) for j in [i[2] for i in all_triples]]
    unique_head = len(set([i[0] for i in all_triples]))
    unique_reln = len(set([i[1] for i in all_triples]))
    unique_tail = len(set([i[2] for i in all_triples]))
    proportion_of_i_head = 100 * len([i[0] for i in all_triples if i[0] == 'i']) / len(all_triples)
    print("n: ", len(all_sentences))
    print("mean words: ", np.mean(words_in_sentence))
    print("std dev words: ", np.std(words_in_sentence))
    print("number of unique head\_entity: ", unique_head)
    print("proportion of i: ", proportion_of_i_head, "%")
    print("number of unique reln\_entity: ", unique_reln)
    print("number of unique tail\_entity: ", unique_tail)
    print("mean head words: ", np.mean(words_in_head_entity))
    print("std dev tail words: ", np.std(words_in_head_entity))
    print("mean tail words: ", np.mean(words_in_tail_entity))
    print("std dev tail words: ", np.std(words_in_tail_entity))

def save_to_csv(sentences, triples, csv_name):
    heads = [i[0] for i in triples]
    relns = [i[1] for i in triples]
    tails = [i[2] for i in triples]
    data = {
            "ground_head":heads,
            "ground_reln":relns,
            "ground_tail":tails,
            "ground_sentence":sentences
            }

    df= pd.DataFrame.from_dict(data)
    df.to_csv(csv_name)

if __name__ == "__main__":

    datasets = ["train", "dev", "test"]

    all_new_sentences = []
    all_new_triples = []
    all_rejected_sentences = []
    all_rejected_triples = []


    for dataset in datasets:
        print("Dataset: {}".format(dataset))
        filename = "../../dnli/dialogue_nli/dialogue_nli_{}.jsonl".format(dataset)
        all_sentences, all_triples = DNLIDataset.load_sentences_and_triples(filename)
        all_sentences, all_triples = DNLIDataset.remove_duplicates(all_sentences, all_triples)
        all_sentences, all_triples = DNLIDataset.remove_invalid_triples(all_sentences, all_triples)
        all_triples = DNLIDataset.remove_tail_prefix_numbers(all_triples)
        all_triples = DNLIDataset.unify_ontology(all_triples)
        new_sentences, new_triples, rejected_sentences, rejected_triples = DNLIDataset.retain_triples_found_in_sentences_only(all_sentences, all_triples)
#        print(len(new_sentences))
#        print(len(rejected_sentences))
        all_new_sentences += new_sentences
        all_new_triples += new_triples
        all_rejected_sentences += rejected_sentences
        all_rejected_triples += rejected_triples
        
        ## To save for analysis
#        save_to_csv(rejected_sentences, rejected_triples, "../../preprocessed_datasets/{}_sentence_not_in_sentence.csv".format(dataset))
#        save_to_csv(new_sentences, new_triples, "../../preprocessed_datasets/{}_sentence_in_sentence.csv".format(dataset))

        
        
        ## Preprocess baseline model input
        '''
        # WDec and PNDec Extraction
        folder = "../../nyt_format_clean"
        process_sentence_into_nyt_format(new_sentences, new_triples, "{}/{}".format(folder,dataset), system="nyt")
        # WDec and PNDec Inference
        folder = "../../nyt_inference_format_clean"
        process_sentence_into_nyt_format(rejected_sentences, rejected_triples, "{}/{}".format(folder,dataset), system="nyt_inference")
        # DyGIE++
        folder = "../../dygiepp_clean_format"
        process_sentence_into_ace_format(new_sentences, new_triples, "{}/{}.json".format(folder, dataset), system="dygiepp")
        '''
        
    
    get_descriptive_stats(all_new_sentences, all_new_triples)
    get_descriptive_stats(all_rejected_sentences, all_rejected_triples)
    
    sentence_to_triple = {}
    all_all_sentences = all_new_sentences+all_rejected_sentences
    all_all_triples = all_new_triples + all_rejected_triples

    for i in range(len(all_all_sentences)):
        sentence = all_all_sentences[i]
        triple = all_all_triples[i]
        sentence_to_triple[sentence] = triple

    ## To generate sentence_to_triple.json for convai2-rev
    #save_json(sentence_to_triple, "sentence_to_triple.json")

    
    
    
    
    relations = ['[attend_school]',
                 '[dislike]',
                 '[employed_by_company]',
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

    #similar relations
    '''
    like_food, favorite_food
    like_drink, favorite_drink
    like_watching, favorite_show
    like_sports, favorite_sport
    like_read, favorite_book
    like_goto, favorite_place
    like_movie, favorite_movie
    like_music, favorite_music
    
    has_hobby, favorite_hobby, like_activity, favorite_activity
    
    have_sibling, have_chidren (misspell intentional) --> have_family
    
    remove have, not_have, other, like_general, want , dislike, favorite,
    
    remove any tails containing <blank>, unspecified or unknown
    
    Tail containing loads of numbers 
    1. like_animal/has_pet
    2. marital status
    3. have family/ has children
    
    remove all prefix numbers (some tail node are made of only numbers like has_age and that's fine)
    
    
    '''
    


        
               
            
        
    
    
