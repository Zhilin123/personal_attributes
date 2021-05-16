#!/bin/sh
if [ $2 == 'inference' ]
then
  python model/generator_model.py --load_trained_model False --debug_mode False --config_name $1 --lr 2.5e-3 --mode head_reln_tail --data_subset not_within_sentence
  python model/generator_model.py --load_trained_model True --debug_mode False --config_name $1 --mode head_reln_tail --data_subset not_within_sentence --inference_only True --generation_name tail_spans_associated_with_relation+all_ten
  python model/generator_model.py --load_trained_model True --debug_mode False --config_name $1 --mode head_reln_tail --data_subset not_within_sentence --inference_only True --generation_name tail_spans_associated_with_relation+all_ten --generate_train True
  python model/generator_model.py --load_trained_model True --debug_mode False --config_name $1 --mode head_reln_tail --data_subset not_within_sentence --inference_only True --generation_name tail_spans_associated_with_relation+all_ten --generate_test True
  python model/reranker_model.py --load_trained_model False --debug_mode False --config_name $1 --lr 5e-6 --mode head_reln_tail --epochs 8 --inference_only False --model_name bert --train_dataset_filename train_tokens_epoch_tail_spans_associated_with_relation+all_ten.csv
  python model/reranker_model.py --load_trained_model True --debug_mode False --config_name $1 --lr 5e-6 --mode head_reln_tail --epochs 8 --inference_only True --model_name bert --train_dataset_filename train_tokens_epoch_tail_spans_associated_with_relation+all_ten.csv --generate_test True
else
  python model/generator_model.py --load_trained_model False --debug_mode False --config_name $1 --lr 7.5e-4 --mode head_reln_tail --data_subset within_sentence
  python model/generator_model.py --load_trained_model True --debug_mode False --config_name $1 --mode head_reln_tail --data_subset within_sentence --inference_only True --generation_name original_spans+all_ten
  python model/generator_model.py --load_trained_model True --debug_mode False --config_name $1 --mode head_reln_tail --data_subset within_sentence --inference_only True --generation_name original_spans+all_ten --generate_train True
  python model/generator_model.py --load_trained_model True --debug_mode False --config_name $1 --mode head_reln_tail --data_subset within_sentence --inference_only True --generation_name original_spans+all_ten --generate_test True
  python model/reranker_model.py --load_trained_model False --debug_mode False --config_name $1 --lr 5e-6 --mode head_reln_tail --epochs 8 --inference_only False --model_name bert --train_dataset_filename train_tokens_epoch_original_spans+all_ten.csv
  python model/reranker_model.py --load_trained_model True --debug_mode False --config_name $1 --lr 5e-6 --mode head_reln_tail --epochs 8 --inference_only True --model_name bert --train_dataset_filename train_tokens_epoch_original_spans+all_ten.csv --generate_test True
