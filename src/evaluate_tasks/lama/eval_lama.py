import argparse
from argparse import ArgumentParser
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict
import sys
sys.path.append('../')
from lama.lama_utils import load_file
from lama.model import Roberta
from lama.batch_eval_KB_completion import run_evaluation

common_vocab_path = "data/common_vocab_cased.txt"
# model_path = "../model/"
# model_path = r"/home/gzcheng/Projects/mop/src/knowledge_infusion/relation_prompt/checkpoints/roberta-base_20211029_165112_adapter"
# model_path = r"/home/gzcheng/Projects/mop/src/knowledge_infusion/relation_prompt/model_dirroberta-base_20211028_181648_adapter"
# model_path = r"/home/gzcheng/Projects/mop/src/knowledge_infusion/relation_prompt/checkpoints/roberta-base_20211029_203027_adapter"
model_path = r"/home/gzcheng/Projects/mop/src/knowledge_infusion/relation_prompt/checkpoints/roberta-base_20211030_223707_adapter"

def get_args():
    parser = ArgumentParser(description="Evaluate model on LAMA.")
    parser.add_argument(
        "--train_mode",
        default="fusion",
        type=str,
        required=True,
        help="three modes: fusion, adapter, base",
    )
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--base_model", default=None, type=str, required=True)
    parser.add_argument("--tokenizer", default=None, type=str, required=False)
    parser.add_argument("--cuda", action="store_true", help="to use gpu")
    parser.add_argument("--amp", action="store_true", help="use auto mixed precision")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeat_runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrain_epoch", type=int, default=50)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="t=1: softmax fusion, 0<t<1: gumbel softmax fusion, t<0: MOE",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1,
        help="training examples ratio to be kept.",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--groups", type=str, default=None, help="groups to be chosen")

    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--train_file", default="train.tsv")
    parser.add_argument("--dev_file", default="dev.tsv")
    parser.add_argument("--test_file", default="test.tsv")

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )

    args = parser.parse_args()
    return args

def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre = "data/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def eval_model(relations, data_path_pre, data_path_post):
    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    for relation in relations:
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": None,
            "template": "",
            "batch_size": 64,
            "max_sentence_length": 100,
            "threads": -1,
            "model_path": model_path,
            "pretrain_epoch": 0,
            "base_model": 'roberta-base',
            "temperature": 1,
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]

        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        model = Roberta(args)
        print("Model: {}".format(model.__class__.__name__))
        param_optimizer = list(model.named_parameters())
        for n, p in param_optimizer:
            print(n)
        Precision1 = run_evaluation(args, shuffle_data=False, model=model)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ mean P@1: {}".format(mean_p1))

    for t, l in type_Precision1.items():
        print(
            "@@@ ",
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return mean_p1, all_Precision1


if __name__ == "__main__":
    print("1. Google-RE")
    parameters = get_GoogleRE_parameters()
    eval_model(*parameters)

    print("2. T-REx")
    parameters = get_TREx_parameters()
    eval_model(*parameters)
