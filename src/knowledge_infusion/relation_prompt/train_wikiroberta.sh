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