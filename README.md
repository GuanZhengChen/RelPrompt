# Codes for paper: 
> [Zaiqiao Meng, Fangyu Liu, Thomas Hikaru Clark, Ehsan Shareghi, Nigel Collier. Mixture-of-Partitions: Infusing Large Biomedical Knowledge Graphs into BERT. EMNLP2021](https://arxiv.org/abs/2109.04810)

## File structure

- `data_dir`: downstream task dataset used in the experiments.
- `kg_dir`: folder to save the knowledge graphs as well as the partitioned files.
- `model_dir`: folder to save pre-trained models.
- `src`: source code.
  - `evaluate_tasks`: codes for the downstream tasks.
  -  `knowledge_infusion`: knowledge infusion main codes.

## Installation

The code is tested with python 3.8.5, torch 1.7.0 and huggingface transformers 3.5.0. Please view requirements.txt for more details.

## Datasets
- Wikidata5m

## Train knowledge fusion and downstream tasks

### Train Knowledge Infusion
To train knowledge infusion, you can run the following command (see train_wikiroberta.sh) in the knowledge_infusion/knowledge_infusion folder.
```shell
MODEL="roberta-base"
TOKENIZER="roberta-base"
INPUT_DIR="/home/gzcheng/Projects/mop/kg_dir/wikidata5m_alias"
OUTPUT_DIR="checkpoints"
DATASET_NAME="wikidata5m_alas"
ADAPTER_NAMES="entity_predict"
PARTITION=256

python run_pretrain.py \
--model $MODEL \
--tokenizer $TOKENIZER \
--input_dir $INPUT_DIR \
--output_dir $OUTPUT_DIR \
--n_partition $PARTITION \
--use_adapter \
--non_sequential \
--adapter_names  $ADAPTER_NAMES\
--amp \
--cuda \
--num_workers 32 \
--max_seq_length 64 \
--batch_size 64  \
--lr 1e-04 \
--epochs 2 \
--save_step 2000
```
### Train Downstream Tasks
To evaluate the model on a downstream task, you can go to the task folder and see the *.sh file for an example. For example, the following command is used to train a model on pubmedqa dataset over different shuffle_rates.
```shell
MODEL="roberta-base"
TOKENIZER="roberta-base"
ADAPTER_NAMES="entity_predict"
PARTITION=20
shuffle_rates=(0.10 0.20 0.40 0.80 1.00)

for shuffle_rate in ${shuffle_rates[*]}; do
    python run_pretrain.py \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --n_partition $PARTITION \
    --use_adapter \
    --non_sequential \
    --adapter_names  $ADAPTER_NAMES\
    --amp \
    --cuda \
    --shuffle_rate $shuffle_rate \
    --num_workers 32 \
    --max_seq_length 64 \
    --batch_size 256 \
    --bi_direction \
    --lr 1e-04 \
    --epochs 2 \
    --save_step 2000
done
```