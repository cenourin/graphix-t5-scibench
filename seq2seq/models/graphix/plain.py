#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plain (no-RGAT) model wrapper -- used for the t5-small ablation baseline.

TokenizedDataset (seq2seq/utils/dataset_graph.py) still needs graph_pedia to
reconstruct the serialized-schema input text (`new_struct_in`, produced by the
subword/graph preprocessing pipeline), so graph_idx keeps flowing through the
data pipeline unchanged even here -- what differs from rgat.py::Model is that
this class never builds a graph_batch or touches DGL/RGAT at all: it just
loads a standard (non-RGAT-injected) T5ForConditionalGeneration and ignores
graph_idx/graph_nodes_subwords_idx in forward()/generate(). This is the
"plain T5" side of the RGAT-vs-no-RGAT ablation.
"""

from torch import nn
from transformers import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers import PreTrainedModel
import torch
import pdb


class Model(PreTrainedModel):
    def __init__(self, tokenizer, model_cls_wrapper, model_args, config, graph_pedia=None, graph_pedia_eval=None):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.pretrain_model = model_cls_wrapper(AutoModelForSeq2SeqLM).from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        self.config = self.pretrain_model.config

    def gradient_checkpointing_enable(self):
        self.pretrain_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.pretrain_model.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        # kwargs carries graph_idx (and possibly task_ids) from TokenizedDataset --
        # deliberately unused here, this is the no-graph-structure baseline.
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
        ).loss
        if torch.isnan(loss).sum() != 0: pdb.set_trace()
        return {'loss': loss}

    def generate(self, input_ids, attention_mask, **kwargs):
        for k in ('graph_idx', 'graph_nodes_subwords_idx', 'task_ids'):
            kwargs.pop(k, None)
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )
        return generated_ids
