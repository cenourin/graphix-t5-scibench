"""Model wrappers on the modern stack (step A6.2b): ports of seq2seq/models/graphix/rgat.py
(RGAT arm) and seq2seq/models/graphix/plain.py (plain arm).

Changes, all interface-level:
  - no model_cls_wrapper / AutoModelForSeq2SeqLM factory: the RGAT arm loads
    graphix_modern.modeling_t5.T5ForConditionalGeneration, the plain arm the stock
    transformers T5ForConditionalGeneration (as the legacy plain arm did with 4.17's);
    PICARD, which wrapped that factory, is not used by the study;
  - graph_batch edges come from the graph store's relation ids (GraphStore 'rel_ids'),
    the values the legacy code derived per step from the relation names;
  - a NaN loss raises instead of stopping in pdb.set_trace() (unusable in a container);
  - gradient_checkpointing_enable accepts transformers 4.57's gradient_checkpointing_kwargs,
    defaulting to reentrant checkpointing, which is what torch 1.9 / the legacy fork used;
  - _tied_weights_keys declared so the wrapper's checkpoints save as safetensors.
"""
import torch
from transformers import PreTrainedModel

from seq2seq.models.graphix.constants import GRAPHIX_RELATIONS

_TIED = ["pretrain_model.encoder.embed_tokens.weight", "pretrain_model.decoder.embed_tokens.weight",
         "pretrain_model.lm_head.weight"]


class _Wrapper(PreTrainedModel):
    _tied_weights_keys = _TIED

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.pretrain_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {"use_reentrant": True})

    def gradient_checkpointing_disable(self):
        self.pretrain_model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self):
        return self.pretrain_model.is_gradient_checkpointing

    @staticmethod
    def _check(loss):
        if torch.isnan(loss).sum() != 0:
            raise FloatingPointError("NaN loss")
        return {"loss": loss}


class RGATModel(_Wrapper):
    """Port of seq2seq/models/graphix/rgat.py::Model."""

    def __init__(self, tokenizer, model_args, config, graph_store, graph_store_eval=None):
        super().__init__(config)
        from .modeling_t5 import T5ForConditionalGeneration
        self.tokenizer = tokenizer
        self.pretrain_model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir)
        self.config = self.pretrain_model.config
        # train store / eval store, picked by self.training as in the legacy wrapper
        self.graph_pedia = graph_store
        self.graph_pedia_eval = graph_store_eval if graph_store_eval is not None else graph_store
        self.rel2id = {r: i for i, r in enumerate(GRAPHIX_RELATIONS)}

    def graph_factory(self, kwargs):
        graph_idx_batch = kwargs.pop("graph_idx", None)
        device = graph_idx_batch.device
        store = self.graph_pedia if self.training else self.graph_pedia_eval
        batch = []
        for idx in graph_idx_batch.tolist():
            entry = store[int(idx)]
            batch.append({"graph": entry["graph"],
                          "edges": torch.as_tensor(entry["rel_ids"], dtype=torch.long, device=device)})
        return batch

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        graph_batch = self.graph_factory(kwargs)
        loss = self.pretrain_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                                   labels=labels, graph_batch=graph_batch).loss
        return self._check(loss)

    def generate(self, input_ids, attention_mask, **kwargs):
        graph_batch = self.graph_factory(kwargs)
        return self.pretrain_model.generate(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                                            graph_batch=graph_batch, **kwargs)


class PlainModel(_Wrapper):
    """Port of seq2seq/models/graphix/plain.py::Model (stock T5, no graph)."""

    def __init__(self, tokenizer, model_args, config, graph_store=None, graph_store_eval=None):
        super().__init__(config)
        from transformers import T5ForConditionalGeneration
        self.tokenizer = tokenizer
        self.pretrain_model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir)
        self.config = self.pretrain_model.config

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        loss = self.pretrain_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                                   labels=labels).loss
        return self._check(loss)

    def generate(self, input_ids, attention_mask, **kwargs):
        for k in ("graph_idx", "graph_nodes_subwords_idx", "task_ids"):
            kwargs.pop(k, None)
        return self.pretrain_model.generate(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                                            **kwargs)
