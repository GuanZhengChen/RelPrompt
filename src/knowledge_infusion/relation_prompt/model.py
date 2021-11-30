import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification
# from transformers.modeling_bert import BertLayerNorm, gelu
import copy

class RelPrompt(RobertaForMaskedLM):
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, rel=None, devices=None):
        super().__init__(config)
        # self.ent_embeddings = nn.Embedding(num_ent, ent_dim, padding_idx=1)
        # self.ent_embeddings = LargeEmbedding(ip_config, emb_name, ent_lr, num_ent)
        # self.ent_embeddings = nn.Embedding(num_ent, config.hidden_size, padding_idx=1)
        # self.rel_embeddings = nn.Embedding(num_rel, config.hidden_size, padding_idx=1)

        self.prompt_length = len(rel)
        # self.num_prompt = 10
        # self.prompt = nn.Embedding(len(rel), config.hidden_size)
        self.devices = devices
        # rel_emb = copy.deepcopy(self.roberta.embeddings.word_embeddings(torch.tensor(rel)))
        
        # self.prompt = nn.Embedding.from_pretrained(rel_emb)
        self.prompt = nn.Parameter(self.roberta.embeddings.word_embeddings(torch.tensor(rel)).clone())

        self.prompt.requires_grad = True
        # self.ent_lm_head = EntLMHead(config)
        # self.rel_lm_head = RelLMHead(config, num_rel)
        self.apply(self._init_weights)
        # if rel_emb is not None:
        #     self.rel_embeddings = nn.Embedding.from_pretrained(rel_emb, padding_idx=1)
        #     print('pre-trained relation embeddings loaded.')
        # self.tie_rel_weights()

    # def extend_type_embedding(self, token_type=3):
    #     self.roberta.embeddings.token_type_embeddings = nn.Embedding(token_type, self.config.hidden_size,
                                                                    #  _weight=torch.zeros(
                                                                    #      (token_type, self.config.hidden_size)))
    # def extend_attention_mask(self, attention_mask):
    #     extend_mask = torch.ones((attention_mask.shape[0],self.prompt_length)).to(self.config.device)
    #     promt_attention_mask = torch.cat([extend_mask, attention_mask], dim=-1)
    #     return promt_attention_mask


    # def extend_position_id(self, position_id):
    #     extend_position = torch.zeros((position_id.shape[0],self.prompt_length)).to(self.config.device)
    #     promt_position_id = torch.cat([extend_position, position_id], dim=-1)
    #     return promt_position_id

    # def tie_rel_weights(self):
    #     self.rel_lm_head.decoder.weight = self.rel_embeddings.weight
    #     if getattr(self.rel_lm_head.decoder, "bias", None) is not None:
    #         self.rel_lm_head.decoder.bias.data = torch.nn.functional.pad(
    #             self.rel_lm_head.decoder.bias.data,
    #             (0, self.rel_lm_head.decoder.weight.shape[0] - self.rel_lm_head.decoder.bias.shape[0],),
    #             "constant",
    #             0,
    #         )

    def forward(
            self,
            input_ids=None,
            mode='query'
    ):
        # n_word_nodes = n_word_nodes[0]
        # n_entity_nodes = n_entity_nodes[0]
        ent_embedding = self.roberta.embeddings.word_embeddings(input_ids)  # batch x n_word_nodes x hidden_size

        # ent_embeddings = self.ent_embeddings(
        #         input_ids[:, n_word_nodes:n_word_nodes + n_entity_nodes])

        # rel_embeddings = self.rel_embeddings(
        #     input_ids[:, n_word_nodes + n_entity_nodes:])
        # prompt_embed = self.prompt.expand(ent_embedding.shape[0], self.prompt_length, ent_embedding.shape[2]).to(self.devices)
        # promt_embedding = self.prompt[group_idx * self.prompt_length, (group_idx + 1) * self.prompt_length ]
        if mode == 'query':
            prompt_embed = self.prompt.expand(ent_embedding.shape[0], self.prompt_length, ent_embedding.shape[2]).to(self.devices)
            inputs_embeds = torch.cat([ent_embedding[:,:-2,:], prompt_embed, ent_embedding[:,-2:,:]],dim=1)  # batch x seq_len x hidden_size
        else:
            inputs_embeds = ent_embedding
        # add the adapter_name here
        # how to deal with position_ids? set all as 0 or random position
        # prompt_attention_mask = self.extend_attention_mask(attention_mask)
        # promt_position_id = self.extend_position_id(position_ids)
        outputs = self.roberta(
            input_ids=None,
            # attention_mask=prompt_attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=promt_position_id,
            # head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        return outputs
        # sequence_output = outputs[0]  # batch x seq_len x hidden_size

        # loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        # word_logits = self.lm_head(sequence_output)
        # word_predict = torch.argmax(word_logits, dim=-1)
        # masked_lm_loss = loss_fct(word_logits.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        # # ent_cls_weight = self.ent_embeddings(ent_index[0].view(1,-1)).squeeze()
        # # ent_logits = self.ent_lm_head(sequence_output[:, n_word_nodes:n_word_nodes + n_entity_nodes, :],
        # #                               ent_cls_weight)
        # # ent_predict = torch.argmax(ent_logits, dim=-1)
        # # ent_masked_lm_loss = loss_fct(ent_logits.view(-1, ent_logits.size(-1)), ent_masked_lm_labels.view(-1))

        # # rel_logits = self.rel_lm_head(sequence_output[:, n_word_nodes + n_entity_nodes:, :])
        # # rel_predict = torch.argmax(rel_logits, dim=-1)
        # # rel_masked_lm_loss = loss_fct(rel_logits.view(-1, rel_logits.size(-1)), rel_masked_lm_labels.view(-1))
        # loss = masked_lm_loss
        # return {'loss': loss,
        #         'word_pred': word_predict
        #         }


# class EntLMHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         # self.dropout = nn.Dropout(p=dropout)

#     def forward(self, features, weight, **kwargs):
#         x = self.dense(features)
#         x = F.gelu(x)
#         # x = self.dropout(x)
#         x = self.layer_norm(x)
#         x = x.matmul(weight.t())

#         return x


# class RelLMHead(nn.Module):
#     def __init__(self, config, num_rel):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#         self.decoder = nn.Linear(config.hidden_size, num_rel, bias=False)
#         self.bias = nn.Parameter(torch.zeros(num_rel), requires_grad=True)
#         # self.dropout = nn.Dropout(p=dropout)

#         self.decoder.bias = self.bias

#     def forward(self, features, **kwargs):
#         x = self.dense(features)
#         x = F.gelu(x)
#         # x = self.dropout(x)
#         x = self.layer_norm(x)

#         x = self.decoder(x)

#         return x
