import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tabulate import tabulate
from transformers import (
    AdamW,
    AdapterConfig,
    AdapterType,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    RobertaTokenizer
)

import wandb
# from utils.bert_trainer import BertTrainer
from utils.bert_trainer_prompt import BertTrainer
from utils.common import print_args_as_table
from utils.kg_processor import KGProcessor, KGProcessor_rel, KGProcessor_prompt
from model import RelPrompt

# 1. Start a W&B run
wandb.init(project="Entity prediction with partition")


from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Relation and Entity prediction.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="use GPU?",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--model", default="roberta-base", type=str)
    parser.add_argument("--tokenizer", default=None, type=str, required=False)
    parser.add_argument("--subset", default=1.0, type=float, required=False)
    parser.add_argument("--use_adapter", action="store_true", help="use adapters?")
    parser.add_argument("--shuffle_rate", type=str, default=None)
    parser.add_argument(
        "--cache_token_encodings",
        action="store_true",
        help="use cached tokenized encodings?",
    )
    parser.add_argument(
        "--non_sequential",
        action="store_true",
        help="if true, will initial a new model for each group",
    )
    parser.add_argument("--adapter_names", default=None, type=str, required=False)
    parser.add_argument(
        "--input_dir",
        default="/home/gzcheng/Projects/mop/kg_dir/wikidata5m_alias",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="models/kg_bert/",
        type=str,
        required=False,
    )
    parser.add_argument("--trained_model", default=None, type=str)
    parser.add_argument("--amp", action="store_true", help="use auto mixed precision")
    parser.add_argument("--save_step", default=2000, type=int, required=False)
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for",
    )
    parser.add_argument(
        "--CRate",
        default=8,
        type=int,
        help="adapter_reduction_factor, #{2,16,64}",
    )
    parser.add_argument(
        "--n_partition",
        default=50,
        type=int,
        help="Number of groups when partitioning graph",
    )
    parser.add_argument(
        "--sub_group_idx",
        default=None,
        type=int,
        help="Index of sub-groups of certain partitions",
    )
    parser.add_argument(
        "--bi_direction",
        action="store_true",
        help="Do bi-direction prediction for both head and tail nodes?",
    )
    parser.add_argument(
        "--adapter_layers",
        default=None,
        type=str,
        help="layers string for deploying adapters,e.g. 1,2,3, if None, will deploy adapters in all the layers",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    args = parser.parse_args()
    return args



def init_model(args, relid=None):
    print(f"Initializing model from {args.model}")
    from_tf = False
    if (
        (
            ("BioRedditBERT" in args.model)
            or ("BioBERT" in args.model)
            or ("SapBERT" in args.model)
        )
        and "step_" not in args.model
        and "epoch_" not in args.model
    ):
        from_tf = True
    config = AutoConfig.from_pretrained(args.model)
    # model = AutoModel.from_pretrained(
    #     args.model, from_tf=from_tf, config=config
    # )
    model = RelPrompt.from_pretrained(
        args.model, config=config, rel=relid, devices=args.device
    )
    if args.use_adapter:
        adapter_config = AdapterConfig.load(
            "pfeiffer",
            # non_linearity=adapter_args.adapter_non_linearity,
            reduction_factor=args.CRate,  # adapter_args.adapter_reduction_factor, #{2,16,64}
            leave_out=[]
            if args.adapter_layers is None
            else list(
                set(range(model.config.num_hidden_layers)) - set(args.adapter_layers)
            ),
        )
        model.add_adapter(
            args.adapter_names, config=adapter_config
        )
        model.train_adapter([args.adapter_names])
    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    for n, p in param_optimizer:
        if p.requires_grad:
            print(n)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        weight_decay=0.01,
        correct_bias=False,
    )
    return model, optimizer

# def build_rel(path):
#     triples = os.path.join(path, 'wikidata5m_transductive_train.txt')
#     rel_dict = 
#     for triple in triples:


if __name__ == "__main__":
    # Set default configuration in args.py
    args = get_args()
    # args.cuda = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # device = 'cuda'
    n_gpu = torch.cuda.device_count() if args.cuda else 0
    model_str = args.model
    if "/" in model_str:
        model_str = model_str.split("/")[1]
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.model_str = f"{model_str}_{timestamp_str}"
    if args.use_adapter:
        args.model_str += "_adapter"
    args.save_path = args.output_dir + '/' + args.model_str
    os.makedirs(args.save_path, exist_ok=True)

    print("Device:", str(device).upper())
    print("Number of GPUs:", n_gpu)
    print_args_as_table(args)
    # Set random seed for reproducibility
    if args.seed is None:
        args.seed = int(time.time())
        print(f"generate random seed {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    data_processor = KGProcessor_prompt(
        args.input_dir,
        args.subset,
        n_partition=args.n_partition,
        bi_direction=args.bi_direction,
        sub_group_idx=args.sub_group_idx,
        shuffle_rate=args.shuffle_rate,
    )
    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    args.is_multilabel = False
    args.is_hierarchical = False
    args.adapter_layers = (
        None
        if args.adapter_layers == None
        else [int(i) for i in args.adapter_layers.split(",")]
    )
    if args.tokenizer is None:
        args.tokenizer = args.model
    wandb.config.update(args)
    # print(model)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # model, optimizer = init_model(args)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer)
    # rel = data_processor.id2rel[data_processor.top_rel]
    rel_names = list(map(data_processor.id2rel.get, data_processor.top_rel))
    relations = tokenizer(rel_names, add_special_tokens=False, add_prefix_space=True)['input_ids']
    model, optimizer = init_model(args, relations[0])
    for group_idx in range(args.n_partition):
        if group_idx != 0 and args.non_sequential:
            model, optimizer = init_model(args, relations[group_idx])
        wandb.watch(model)
        trainer = BertTrainer(model, optimizer, data_processor, tokenizer, args)
        # if n_gpu > 1:
        #     model.module.classifier = nn.Linear(
        #         in_features=768,
        #         out_features=data_processor.num_class_list[group_idx],
        #         bias=True,
        #     )
        #     model.module.classifier.to(device)
        # else:
        #     model.classifier = nn.Linear(
        #         in_features=768,
        #         out_features=data_processor.num_class_list[group_idx],
        #         bias=True,
        #     )
        #     model.classifier.to(device)
        if args.cache_token_encodings:
            trainer.train_subgraph_cache_tokens(group_idx)
        else:
            trainer.train_subgraph(group_idx)
