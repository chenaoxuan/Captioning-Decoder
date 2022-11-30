# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from basemodel import PreTrainedModel
from decoder import DecoderModel

Config = {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "type_vocab_size": 2,
    "vocab_size": 30522,
    "num_decoder_layers": 1,
    "max_target_embeddings": 512
}


class Decoder(UniVLPreTrainedModel):
    def __init__(self, decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight, task_config):
        super(Decoder, self).__init__(decoder_config)
        self.task_config = task_config
        assert self.task_config.max_words <= decoder_config.max_target_embeddings

        self.decoder = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight)
        self.decoder_loss_fct = CrossEntropyLoss(ignore_index=-1)

        self.apply(self.init_weights)

    def forward(self, attention_mask, video_mask, cross_output,
                input_caption_ids, decoder_mask, output_caption_ids):

        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
        decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        if self.training:
            decoder_scores = self._get_decoder_score(attention_mask, video_mask, cross_output,
                                                     input_caption_ids, decoder_mask)
            output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
            decoder_loss = self.decoder_loss_fct(decoder_scores.view(-1, self.bert_config.vocab_size),
                                                 output_caption_ids.view(-1))
            return decoder_loss
        else:
            return None

    def _get_decoder_score(self, attention_mask, video_mask,
                           cross_output, input_caption_ids, decoder_mask):
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        decoder_scores = self.decoder(input_caption_ids, encoder_outs=cross_output, answer_mask=decoder_mask,
                                      encoder_mask=concat_mask)
        return decoder_scores

    # 后期的beam search要用，暂时保留
    # def decoder_caption(self, attention_mask, video_mask, cross_output, input_caption_ids,
    #                     decoder_mask, shaped=False, get_logits=False):
    #     if shaped is False:
    #         attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
    #         video_mask = video_mask.view(-1, video_mask.shape[-1])
    #         input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
    #         decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])
    #
    #     decoder_scores = self._get_decoder_score(attention_mask, video_mask, cross_output,
    #                                              input_caption_ids, decoder_mask, shaped=True)
    #
    #     if get_logits:
    #         return decoder_scores
    #
    #     _, decoder_scores_result = torch.max(decoder_scores, -1)
    #
    #     return decoder_scores_result


############################################
#################Evaluation#################
############################################