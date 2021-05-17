This project seeks to extract and infer personal attributes from dialogue

## Dependencies

```sh
pip install -r requirements.txt
```
Python3 is required

GPU (>= 16GB memory) is highly recommended.

## Dataset

Dataset has been repurposed from [DialogNLI](https://wellecks.github.io/dialogue_nli/)

Please put the unzipped ```dnli``` folder at the same level as the ```src``` folder. ```dnli``` folder should contain ```dialogue_nli/dialogue_nli_dev.jsonl``` , ```dialogue_nli/dialogue_nli_train.jsonl``` and ```dialogue_nli/dialogue_nli_test.jsonl```

## Benefits to Open-domain Chit-chat (PersonaChat)

First install ParlAI from source, at the same level at src

```sh
git clone https://github.com/facebookresearch/ParlAI.git ParlAI
cd ParlAI
python setup.py develop
```

Create a parlai_internal folder and copy convai2-rev into parlai_internal/tasks

```sh
cp example_parlai_internal parlai_internal
cp ../src/convai2-rev parlai_internal/tasks/convai2-rev
```

Train a model using the new task

```sh
cd parlai
parlai train_model -mf <working_directory>/model -m transformer/generator -im zoo:blender/blender_90M/model -vp 15 -t internal:convai2-rev:normalized -bs 32 -ltim 60 --rank-candidates True --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 True --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 --skip-generation False -stim 6000 -vme 20000 -bs 16 -vmt hits@1 -vmm max --save-after-valid True
```
## Training + Testing

The command below trains the entire GenRe model.

```sh
# <task> {extraction, inference}
sh train.sh <working_directory> <task>
```

## Analysis

To show distribution of dependency labels and POS tags in the Extraction dataset

```sh
cd analysis
tar -xvf data.tar.gz

python -m spacy download en_core_web_trf

python preprocess_linguistic_analysis.py --debug_mode False \
--csv_filename data/eval_tokens_within-sentence.csv

# <interested_field> {dependency_labels, big_pos_tags}
python linguistic_analysis.py --interested_field  <interested_field> \
--csv_filename data/eval_tokens_within-sentence_analysis.csv

```

To show how tail entities can be linked to sentences after various transformation in the Inference dataset

```sh
#Call conceptnet API to obtain words linked by commonsense
python call_and_save_conceptnet_api.py --category_of_words sentence --dataset eval --field_of_interest related --subset all
python call_and_save_conceptnet_api.py --category_of_words sentence --dataset eval --field_of_interest connected --subset all
python call_and_save_conceptnet_api.py --category_of_words tail --dataset eval --field_of_interest related --subset all
python call_and_save_conceptnet_api.py --category_of_words tail --dataset eval --field_of_interest connected --subset all

# run analysis
python tail_entity_not_within_sentence_analysis.py --mode dataset_analysis
```

To show the proportion of predicted tail entities can be linked to sentences after various transformation in the Inference dataset

```sh
#Call conceptnet API to obtain words linked by commonsense
python call_and_save_conceptnet_api.py --category_of_words sentence --dataset eval --field_of_interest related --subset all
python call_and_save_conceptnet_api.py --category_of_words sentence --dataset eval --field_of_interest connected --subset all

python tail_entity_not_within_sentence_analysis.py --mode prediction_analysis
python tail_entity_proportion.py
```
