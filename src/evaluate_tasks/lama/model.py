import os
import torch
import torch.nn as nn
import numpy as np
from fastNLP import seq_len_to_mask
from transformers import RobertaForMaskedLM, RobertaTokenizer, BertPreTrainedModel, RobertaModel, RobertaConfig
# from transformers.modeling_roberta import RobertaLMHead
from transformers import (
    AdamW,
    AdapterConfig,
    AdapterFusionConfig,
    AutoConfig,
    AutoTokenizer,
)
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule
from os import listdir

import sys
# sys.path.append(os.path.dirname(__file__) + os.sep + '../')
sys.path.append('../')
sys.path.append('.../')
sys.path.append(r'/home/gzcheng/Projects/mop/src')
from knowledge_infusion.relation_prompt.model import RelPrompt

class Roberta(object):

    def __init__(self, args):
        # self.dict_file = "{}/{}".format(args.roberta_model_dir, args.roberta_vocab_name)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        if args.model_path is not None:
            print("Testing CoLAKE...")
            print('loading model parameters from {}...'.format(args.model_path))
            # config = RobertaConfig.from_pretrained('roberta-base', type_vocab_size=3)
            # self.model = RobertaForMaskedLM(config=config)
            # states_dict = torch.load(os.path.join(args.model_path, 'model.bin'))
            # self.model.load_state_dict(states_dict, strict=False)
            config, self.model = self.load_fusion_adapter_model(args)
        else:
            print("Testing RoBERTa baseline...")
            self.model = RobertaForMaskedLM.from_pretrained('roberta-base')

        self._build_vocab()
        self._init_inverse_vocab()
        self._model_device = 'cpu'
        self.max_sentence_length = args.max_sentence_length

    def _cuda(self):
        self.model.cuda()

    def _build_vocab(self):
        self.vocab = []
        for key in range(len(self.tokenizer)):
            value = self.tokenizer.decode([key])
            if value[0] == " ":  # if the token starts with a whitespace
                value = value.strip()
            else:
                # this is subword information
                value = "_{}_".format(value)

            if value in self.vocab:
                # print("WARNING: token '{}' is already in the vocab".format(value))
                value = "{}_{}".format(value, key)

            self.vocab.append(value)
        print("size of vocabulary: {}".format(len(self.vocab)))

    def _init_inverse_vocab(self):
        self.inverse_vocab = {w: i for i, w in enumerate(self.vocab)}

    def try_cuda(self):
        """Move model to GPU if one is available."""
        if torch.cuda.is_available():
            if self._model_device != 'cuda':
                self._cuda()
                self._model_device = 'cuda'
        else:
            print('No CUDA found')

    def init_indices_for_filter_logprobs(self, vocab_subset):
        index_list = []
        new_vocab_subset = []
        for word in vocab_subset:
            if word in self.inverse_vocab:
                inverse_id = self.inverse_vocab[word]
                index_list.append(inverse_id)
                new_vocab_subset.append(word)
            else:
                msg = "word {} from vocab_subset not in model vocabulary!".format(word)
                print("WARNING: {}".format(msg))

        indices = torch.as_tensor(index_list)
        return indices, index_list

    def filter_logprobs(self, log_probs, indices):
        new_log_probs = log_probs.index_select(dim=2, index=indices)
        return new_log_probs

    def get_id(self, input_string):
        # Roberta predicts ' London' and not 'London'
        string = " " + str(input_string).strip()
        tokens = self.tokenizer.encode(string, add_special_tokens=False)
        # return [element.item() for element in tokens.long().flatten()]
        return tokens

    def get_batch_generation(self, samples_list, try_cuda=True):
        if not samples_list:
            return None
        if try_cuda:
            self.try_cuda()

        tensor_list = []
        masked_indices_list = []
        max_len = 0
        output_tokens_list = []
        seq_len = []
        for sample in samples_list:
            masked_inputs_list = sample["masked_sentences"]

            tokens_list = [self.tokenizer.bos_token_id]

            for idx, masked_input in enumerate(masked_inputs_list):
                tokens_list.extend(self.tokenizer.encode(" " + masked_input.strip(), add_special_tokens=False))
                tokens_list.append(self.tokenizer.eos_token_id)

            # tokens = torch.cat(tokens_list)[: self.max_sentence_length]
            tokens = torch.tensor(tokens_list)[: self.max_sentence_length]
            output_tokens_list.append(tokens.long().cpu().numpy())

            seq_len.append(len(tokens))
            if len(tokens) > max_len:
                max_len = len(tokens)
            tensor_list.append(tokens)
            masked_index = (tokens == self.tokenizer.mask_token_id).nonzero().numpy()
            for x in masked_index:
                masked_indices_list.append([x[0]])
        tokens_list = []
        for tokens in tensor_list:
            pad_lenght = max_len - len(tokens)
            if pad_lenght > 0:
                pad_tensor = torch.full([pad_lenght], self.tokenizer.pad_token_id, dtype=torch.int)
                tokens = torch.cat((tokens, pad_tensor.long()))
            tokens_list.append(tokens)

        batch_tokens = torch.stack(tokens_list)
        seq_len = torch.LongTensor(seq_len)
        attn_mask = seq_len_to_mask(seq_len)

        with torch.no_grad():
            # with utils.eval(self.model.model):
            self.model.eval()
            outputs = self.model(
                batch_tokens.long().to(device=self._model_device),
                attention_mask=attn_mask.to(device=self._model_device)
            )
            log_probs = outputs[0]

        return log_probs.cpu(), output_tokens_list, masked_indices_list

    def load_fusion_adapter_model(self, args):
        """Load fusion adapter model.

        Args:
            args ([type]): [description]

        Returns:
            [type]: [description]
        """
        adapter_names_dict = self.search_adapters(args)
        base_model = RelPrompt.from_pretrained(
            args.base_model
        )
        fusion_adapter_rename = []
        for model_path, adapter_names in adapter_names_dict.items():
            for adapter_name in adapter_names:
                adapter_dir = os.path.join(model_path, adapter_name)
                new_adapter_name = model_path[-14:][:-8] + "_" + adapter_name
                # new_adapter_name = adapter_name
                base_model.load_adapter(adapter_dir, load_as=new_adapter_name)
                # print(f"Load adapter:{new_adapter_name}")
                fusion_adapter_rename.append(new_adapter_name)
        fusion_config = AdapterFusionConfig.load("dynamic", temperature=args.temperature)
        base_model.add_adapter_fusion(fusion_adapter_rename, fusion_config)
        base_model.set_active_adapters(fusion_adapter_rename)
        config = AutoConfig.from_pretrained(
            os.path.join(adapter_dir, "adapter_config.json")
        )
        # base_model.train_fusion([adapter_names])
        return config, base_model


    # def load_adapter_model(self, args):
    #     model_path = os.path.join(args.model_dir, args.model)
    #     base_model = AutoModelForMultipleChoice.from_pretrained(
    #         args.base_model, from_tf=get_tf_flag(args)
    #     )
    #     adapter_config = AdapterConfig.load(os.path.join(model_path, "adapter_config.json"))
    #     config = AutoConfig.from_pretrained(os.path.join(model_path, "adapter_config.json"))
    #     base_model.load_adapter(model_path, config=adapter_config)
    #     print(f"Load adapter:{config.name}")
    #     base_model.set_active_adapters([config.name])
    #     return config, base_model
    
    def get_tf_flag(self, args):
        from_tf = True
        # if (
        #     (
        #         ("BioRedditBERT" in args.model)
        #         or ("BioBERT" in args.model)
        #         or ("SapBERT" in args.model)
        #     )
        #     and "step_" not in args.model
        #     and "epoch_" not in args.model
        # ):
        #     from_tf = True

        # if ("SapBERT" in args.model) and ("original" in args.model):
        #     from_tf = False
        return from_tf


    def search_adapters(self, args):
        """[Search the model_path, take all the sub directions as adapter_names]

        Args:
            args (ArgumentParser)

        Returns:
            [dict]: {model_path:[adapter_names]}
        """
        adapter_paths_dic = {}
        # if "," in args.model:
        #     for model in args.model.split(","):  # need to fusion from two or more models
        #         model_path = args.model_dir + model
        #         adapter_paths = [f for f in listdir(model_path)]
        #         print(f"Found {len(adapter_paths)} adapter paths")
        #         adapter_paths = self.check_adapter_names(model_path, adapter_paths)
        #         adapter_paths_dic[model_path] = adapter_paths
        # else:
        # model_path = args.model_dir + args.model
        adapter_paths = [f for f in listdir(args.model_path)]
        print(f"Found {len(adapter_paths)} adapter paths")
        adapter_paths = self.check_adapter_names(args.model_path, adapter_paths)
        adapter_paths_dic[args.model_path] = adapter_paths
        return adapter_paths_dic
    
    def check_adapter_names(self, model_path, adapter_names):
        """[Check if the adapter path contrains the adapter model]

        Args:
            model_path ([type]): [description]
            adapter_names ([type]): [description]

        Raises:
            ValueError: [description]
        """
        checked_adapter_names = []
        print(f"Checking adapter namer:{model_path}:{len(adapter_names)}")
        for adapter_name in adapter_names:  # group_0_epoch_1
            adapter_model_path = os.path.join(model_path, adapter_name)
            # if f"epoch_{args.pretrain_epoch}" not in adapter_name:
            if f"epoch_0" not in adapter_name:
                # check pretrain_epoch
                continue
            # if args.groups and int(adapter_name.split("_")[1]) not in set(args.groups):
            #     # check selected groups
            #     continue
            adapter_model_path = os.path.join(adapter_model_path, "pytorch_adapter.bin")
            assert os.path.exists(
                adapter_model_path
            ), f"{adapter_model_path} adapter not found."

            checked_adapter_names.append(adapter_name)
        # print(f"Valid adapters ({len(checked_adapter_names)}):{checked_adapter_names}")
        return checked_adapter_names