#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from transformers import AutoTokenizer
from collections import defaultdict
from .constants import GRAPHIX_RELATIONS, PROMPT_MAPPING
import pickle
import torch
import pdb

from torch import nn
from transformers import AutoTokenizer
from ..modeling_auto import AutoModelForSeq2SeqLM
from transformers import PreTrainedModel
from seq2seq.utils.dataset_graph import get_graph_entry


class Model(PreTrainedModel):
    def __init__(self, tokenizer, model_cls_wrapper, model_args, config, graph_pedia, graph_pedia_eval=None):
        super().__init__(config)

        # Load tokenizer and model.
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
        from ..modeling_t5 import T5ForConditionalGeneration

        # import graph part:
        # graph_pedia is the train-split store; graph_pedia_eval defaults to the same
        # object (Spider's single graph_pedia_total.bin, train+dev merged with dev's
        # graph_idx offset so the two never collide -- see inject_syntax.py). For
        # ScienceBenchmark, train and dev graph_pedia are two SEPARATE files because
        # dev's graph_idx is 0-based, the same range as train's (--dev_graph_idx_offset
        # 0) -- indexing a merged/bare dict by graph_idx alone would silently return
        # the wrong example's graph. self.training (set by the standard nn.Module
        # train()/eval() calls the HF Trainer already makes around each phase) picks
        # the correct store per forward()/generate() call.
        self.graph_pedia = graph_pedia
        self.graph_pedia_eval = graph_pedia_eval if graph_pedia_eval is not None else graph_pedia
        self.rel2id, self.id2rel = self.enumerate_relation(GRAPHIX_RELATIONS)

    def gradient_checkpointing_enable(self):
        # Model is a thin wrapper around self.pretrain_model (the actual RGAT-aware
        # T5ForConditionalGeneration) -- it inherits from the base transformers
        # PreTrainedModel, not from T5PreTrainedModel, so it has no
        # _set_gradient_checkpointing of its own and the default
        # PreTrainedModel.gradient_checkpointing_enable()'s self.apply(...) cascade
        # would either raise ("does not support gradient checkpointing") or silently
        # no-op on Model's own (non-T5Stack) submodules without ever reaching
        # pretrain_model's real T5Stack instances. Delegate straight to the wrapped
        # model, whose class already implements this correctly.
        self.pretrain_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.pretrain_model.gradient_checkpointing_disable()


    def enumerate_relation(self, relations):
        word2id = {}
        id2word = {}

        for i, r in enumerate(relations):
            word2id[r] = i
            id2word[i] = r

        return word2id, id2word

    def get_graph(self, kwargs):

        ''' load graph_idx and convert into list '''
        graph_idx_batch = kwargs.pop('graph_idx', None)
        device = graph_idx_batch.device
        graph_idx_batch_lst = [int(idx[0]) for idx in graph_idx_batch]
        '''load graph_node_idx and convert into list'''
        graph_node_idx_batch = kwargs.pop('graph_nodes_subwords_idx', None)

        graph_pedia = self.graph_pedia if self.training else self.graph_pedia_eval
        new_graph_batch = [] # list of dicts
        for i, graph_idx in enumerate(graph_idx_batch_lst):
            new_graph = self.graph_postprocess(get_graph_entry(graph_pedia, graph_idx), device)
            new_graph_batch.append(new_graph)

        return new_graph_batch

    def graph_factory(self, kwargs):

        '''load and postporcess graphs'''
        graph_idx_batch = kwargs.pop('graph_idx', None)
        device = graph_idx_batch.device
        graph_idx_batch_lst = [int(idx) for idx in graph_idx_batch]

        graph_pedia = self.graph_pedia if self.training else self.graph_pedia_eval
        new_graph_batch = []
        for i, graph_idx in enumerate(graph_idx_batch_lst):
            new_graph = self.graph_postprocess(get_graph_entry(graph_pedia, graph_idx), device)
            new_graph_batch.append(new_graph)


        return new_graph_batch


    def graph_postprocess(self, graph: dict, device):
        new_graph = {}
        edges = graph['edges']
        rel_ids = list(map(lambda r: self.rel2id[r[2]], edges))

        new_graph['edges'] = torch.tensor(rel_ids, dtype=torch.long, device=device)
        new_graph['graph'] = graph['graph']
        # new_graph['question_subword_mask'] = torch.tensor(graph['question_subword_mask'], dtype=torch.bool)
        # new_graph['schema_subword_mask'] = torch.tensor(graph['schema_subword_mask'], dtype=torch.bool)

        return new_graph


    def forward(self, input_ids, attention_mask, labels, **kwargs):
        graph_batch = self.graph_factory(kwargs)
        # self.relation_init_prompt(self.rel2id)
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
            graph_batch = graph_batch,
            # relation_embedding = self.relation_embedding
        ).loss
        if torch.isnan(loss).sum() != 0: pdb.set_trace()
        return {'loss': loss}

    def generate(self, input_ids, attention_mask, **kwargs):
        graph_batch = self.graph_factory(kwargs)
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            graph_batch=graph_batch,
            # relation_embedding=self.relation_embedding
            **kwargs,
        )

        return generated_ids