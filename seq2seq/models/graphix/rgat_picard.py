#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import OrderedDict
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


class Model(nn.Module):
    def __init__(self, tokenizer, model_cls_wrapper, model_args, config, graph_pedia):
        super().__init__()

        # Load tokenizer and model.
        # NOTE on model class selection: AutoModelForSeq2SeqLM's model_type->class mapping
        # (auto_factory.py::_LazyAutoMapping._load_attr_from_module) always dynamically
        # imports from the stock `transformers.models.*` package -- it has no path to this
        # project's RGAT-aware T5ForConditionalGeneration (seq2seq/models/modeling_t5.py).
        # The only place that ever swapped in the correct class was a PICARD-specific
        # special case inside picard_model_wrapper.py::with_picard()'s `from_pretrained`
        # override, which only runs when use_picard=True. With use_picard=False,
        # `model_cls_wrapper` is the identity function, so the old code below silently built
        # a plain, non-graph-aware T5 -- confirmed via `py-spy dump` showing initialization
        # running through `transformers/models/t5/modeling_t5.py`, and independently via
        # `_get_model_class(config, AutoModelForSeq2SeqLM._model_mapping)` resolving to the
        # same stock class. That model can never correctly receive this checkpoint's
        # RGAT-specific weights via `load_state_dict()`. We import the graph-aware class
        # directly here instead, which is correct regardless of the PICARD setting.
        # (Scope note: this only fixes model *construction*. The use_picard=True path's
        # PICARD-specific `generate()` override, applied elsewhere via `model_cls_wrapper`,
        # is untouched and still depends on going through `with_picard()`'s `from_pretrained`
        # override -- not exercised by this fix, since today's target is the use_picard=False
        # smoke-test path.)
        #
        # Separately: every weight of the "t5-3b" base model used to be downloaded and loaded
        # here (~10.6GB) only to be immediately overwritten, in full, by the Graphix-3B
        # checkpoint below -- none of it was ever used. We now build the same architecture
        # from `config` alone (`_from_config`, no pretrained weights fetched or read), which
        # skips that download/load entirely. `torch_dtype=WORKING_DTYPE` makes `_from_config`
        # allocate the (randomly-initialized, soon-to-be-overwritten) parameters directly in
        # that dtype, so no fp32 copy of this skeleton model ever exists -- needed because a
        # GTX 1070 (8GB VRAM) cannot hold this ~3.5B parameter model in fp32 (~14GB) at all;
        # `training_args.fp16` alone does not help here, it only enables autocast for the
        # forward pass, not the stored weight dtype. This is a numerically-relevant precision
        # change, done deliberately to make inference possible on constrained VRAM.
        #
        # WORKING_DTYPE is bf16, not fp16, despite both being 16 bits (same memory cost):
        # fp16's dynamic range tops out at ~65504, and this checkpoint's encoder block 19
        # feed-forward layer genuinely produces activations beyond that -- confirmed directly
        # with per-sublayer nan/inf hooks (block[19]'s T5LayerFF hit `inf`, which its RGAT
        # layer then turned into `nan`, contaminating every block after it and the final
        # generation). That's a magnitude problem, not a precision one: computing that same
        # op in fp32 internally and only casting the *result* down to fp16 still overflowed,
        # because the true value doesn't fit in fp16 at all. bf16 has fp32's exponent range
        # (just fewer mantissa bits), so it holds these values natively -- confirmed the GPU
        # supports bf16 tensor ops (functionally, via generic kernels; Pascal has no bf16
        # tensor cores, so no speed benefit, but correctness and memory footprint are what's
        # needed here). The `T5DenseReluDense`/`T5DenseGatedGeluDense`/`lm_head` fp32-upcast
        # patches made earlier for the fp16 attempt are left in place -- harmless, and still
        # a real precision (not just range) improvement over bf16's narrower mantissa.
        WORKING_DTYPE = torch.bfloat16
        from ..modeling_t5 import T5ForConditionalGeneration as GraphixT5ForConditionalGeneration
        from ..modeling_t5 import T5PreTrainedModel as GraphixT5PreTrainedModel
        from ..modeling_t5 import T5Block as GraphixT5Block
        self._T5Block = GraphixT5Block
        self.tokenizer = tokenizer
        # Every parameter this random init touches is about to be overwritten wholesale by
        # `load_state_dict()` below, so skip HF's post-construction reinit pass (`_init_weights`,
        # applied recursively over every module in `post_init()`) -- for a ~3.5B parameter model
        # this was observed (via `py-spy dump`) to burn several minutes of single-threaded CPU
        # doing RNG work whose result is discarded immediately after. torch 1.9 here predates
        # `torch.nn.utils.skip_init`/meta-device construction, so a temporary no-op monkeypatch
        # is the available lever; restored right after regardless of outcome so it can't leak
        # into any other use of this class within the process.
        # Patched on `T5PreTrainedModel`, not `T5ForConditionalGeneration`, because
        # `_init_weights` is defined once on the shared base class (modeling_t5.py:811) and
        # each submodule (T5Stack for the encoder, another for the decoder) is itself a
        # `T5PreTrainedModel` subclass that independently calls `self.post_init()` with `self`
        # bound to *that* submodule -- patching only the top-level class left those calls
        # resolving `_init_weights` via their own MRO back to the original, unpatched method
        # (confirmed via a second `py-spy dump` still showing real init work after the first,
        # narrower patch).
        #
        # That still isn't the whole story: `nn.Linear`/`nn.Embedding` run their OWN default
        # init (e.g. `kaiming_uniform_`) inside their own `__init__`, *before* `post_init()`
        # ever runs -- a third `py-spy dump` caught this construction still parked in
        # `kaiming_uniform_` via `nn.Linear.reset_parameters`. Same story: every one of those
        # values is about to be overwritten by `load_state_dict()`, so it's also patched out
        # for the duration of construction.
        _orig_init_weights = GraphixT5PreTrainedModel._init_weights
        _orig_linear_reset = torch.nn.Linear.reset_parameters
        _orig_embedding_reset = torch.nn.Embedding.reset_parameters
        GraphixT5PreTrainedModel._init_weights = lambda self, module: None
        torch.nn.Linear.reset_parameters = lambda self: None
        torch.nn.Embedding.reset_parameters = lambda self: None
        try:
            self.pretrain_model = GraphixT5ForConditionalGeneration._from_config(
                config,
                torch_dtype=WORKING_DTYPE,
            )
        finally:
            GraphixT5PreTrainedModel._init_weights = _orig_init_weights
            torch.nn.Linear.reset_parameters = _orig_linear_reset
            torch.nn.Embedding.reset_parameters = _orig_embedding_reset
        # Downcast each tensor to WORKING_DTYPE (bf16) as it comes off disk, via
        # `map_location` -- torch invokes this once per storage *during* deserialization, so
        # only one fp32 storage (at most as large as the single biggest parameter, not the
        # whole 14.2GB file) is ever transiently alive at a time; the fp32 original is
        # unreferenced and collectible as soon as the downcast call returns. This replaces an
        # earlier version that called `torch.load()` plain and then cast the resulting dict
        # afterwards, which held the full fp32 dict (~14.2GB) and a growing downcast copy at
        # once and got the container OOM-killed by the cgroup (confirmed, anon-rss ~25GB at
        # kill time).
        def _load_storage_as_working_dtype(storage, location):
            return storage.bfloat16() if storage.dtype in (torch.float32, torch.float64) else storage

        checkpoint_dict = torch.load(model_args.model_name_or_path+"/pytorch_model.bin", map_location=_load_storage_as_working_dtype)

        print(model_args.model_name_or_path+"/pytorch_model.bin")
        new_checkpoint_dict = OrderedDict()

        model_dict = self.pretrain_model.state_dict()
        for key in list(checkpoint_dict.keys()):
            new_checkpoint_dict[key.replace("pretrain_model.", "")] = checkpoint_dict.pop(key)
        del checkpoint_dict
        model_dict.update(new_checkpoint_dict)
        del new_checkpoint_dict

        self.pretrain_model.load_state_dict(model_dict)
        del model_dict
        self.config = self.pretrain_model.config

        # import graph part:
        self.graph_pedia = graph_pedia
        self.rel2id, self.id2rel = self.enumerate_relation(GRAPHIX_RELATIONS)
        # self.pretrain_model.config.update({"graph_batch":None})
        self._offload_enabled = False

    def to(self, *args, **kwargs):
        # HF Trainer calls `model.to(training_args.device)` unconditionally
        # (transformers/trainer.py::_move_model_to_device). A plain `nn.Module.to('cuda')`
        # would try to move all ~3.5B parameters at once -- confirmed to land at
        # ~8002/8192MiB (fp16 weights + CUDA context), leaving no room for anything the
        # forward pass itself needs (the RGAT layer's DGL graph copy OOM'd here on an
        # 8GB card). Intercept the CUDA case and switch to layer-by-layer CPU<->GPU
        # offloading instead of a full transfer; everything else (CPU, dtype-only calls)
        # goes through the normal `nn.Module.to()`.
        device = None
        for a in args:
            if isinstance(a, (str, torch.device)):
                device = torch.device(a)
                break
        if device is None and "device" in kwargs and kwargs["device"] is not None:
            device = torch.device(kwargs["device"])
        if device is not None and device.type == "cuda":
            self._enable_layer_offload(device)
            return self
        return super().to(*args, **kwargs)

    def _enable_layer_offload(self, device):
        """
        Keep every T5Block (24 encoder + 24 decoder, each holding its own RGAT layer on
        the encoder side) resident on CPU; move each one to `device` right before its
        forward and back to CPU right after, via forward hooks, so only ~1-2 blocks'
        worth of weights (~120-135MB each at this model's d_model=1024/d_ff=16384) are
        ever on the GPU at once instead of the whole ~7.1GB of fp16 weights. Everything
        else (embeddings, final layer norm, lm_head) is small and stays resident on GPU.

        The "next block" is prefetched on a dedicated CUDA stream while the current
        block's forward runs on the default stream, so the H2D copy for block i+1
        overlaps with compute for block i instead of stalling it -- `wait_stream` at
        the start of each block's pre-hook is the sync point that guarantees a block's
        weights have actually finished arriving before its forward reads them.
        """
        if self._offload_enabled:
            return
        self._offload_enabled = True
        self._offload_stream = torch.cuda.Stream(device=device)
        T5Block = self._T5Block

        def move_resident(module):
            # Recurse everywhere except into T5Block instances -- those are handled by
            # the per-block hooks below and must stay on CPU until their forward runs.
            if isinstance(module, T5Block):
                return
            for p in module._parameters.values():
                if p is not None:
                    p.data = p.data.to(device)
            for k, buf in list(module._buffers.items()):
                if buf is not None:
                    module._buffers[k] = buf.to(device)
            for child in module.children():
                move_resident(child)
        move_resident(self.pretrain_model)

        def prefetch(block):
            with torch.cuda.stream(self._offload_stream):
                for p in block.parameters(recurse=True):
                    p.data = p.data.to(device, non_blocking=True)

        def offload_to_cpu(block):
            for p in block.parameters(recurse=True):
                p.data = p.data.to("cpu")

        block_groups = []
        if hasattr(self.pretrain_model, "encoder"):
            block_groups.append(list(self.pretrain_model.encoder.block))
        if hasattr(self.pretrain_model, "decoder"):
            block_groups.append(list(self.pretrain_model.decoder.block))

        for group in block_groups:
            for i, blk in enumerate(group):
                nxt = group[i + 1] if i + 1 < len(group) else None
                is_first = i == 0

                def pre_hook(module, input, _nxt=nxt, _first=is_first):
                    if _first:
                        # Nothing was prefetched ahead of the first block of a fresh
                        # encoder/decoder forward pass -- pull it in synchronously.
                        for p in module.parameters(recurse=True):
                            if not p.is_cuda:
                                p.data = p.data.to(device)
                    # Block this block's own compute on the default stream until any
                    # prefetch of *this* block's weights (issued during the previous
                    # block's pre-hook) has actually completed.
                    torch.cuda.current_stream(device).wait_stream(self._offload_stream)
                    if _nxt is not None:
                        prefetch(_nxt)

                def post_hook(module, input, output):
                    offload_to_cpu(module)

                blk.register_forward_pre_hook(pre_hook)
                blk.register_forward_hook(post_hook)


    def enumerate_relation(self, relations):
        word2id = {}
        id2word = {}

        for i, r in enumerate(relations):
            word2id[r] = i
            id2word[i] = r

        return word2id, id2word

    def graph_factory(self, kwargs):

        '''load and postporcess graphs'''
        graph_idx_batch = kwargs.pop('graph_idx', None)
        device = graph_idx_batch.device
        graph_idx_batch_lst = [int(idx) for idx in graph_idx_batch]

        new_graph_batch = []
        for i, graph_idx in enumerate(graph_idx_batch_lst):
            new_graph = self.graph_postprocess(self.graph_pedia[graph_idx], device)
            new_graph_batch.append(new_graph)
        

        return new_graph_batch


    def graph_postprocess(self, graph: dict, device):
        new_graph = {}
        edges = graph['edges']
        rel_ids = list(map(lambda r: self.rel2id[r[2]], edges))

        new_graph['edges'] = torch.tensor(rel_ids, dtype=torch.long, device=device)
        new_graph['graph'] = graph['graph']
        
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
        if torch.isnan(loss).sum() != 0:
            # This was `pdb.set_trace()` in the original code -- an interactive debugger
            # breakpoint left in a path that also runs during non-interactive/automated
            # eval. With no attached TTY (e.g. `docker run -d`), pdb blocks on stdin
            # until EOF, then raises `bdb.BdbQuit` and kills the whole process -- observed
            # directly: a run hung ~21h at this exact line before dying that way. Kept as a
            # print instead of removed: with WORKING_DTYPE == fp16, this NaN was traced to
            # a real range-overflow bug (encoder block 19's FFN exceeding fp16's ~65504 max,
            # silently turned into NaN by the RGAT layer immediately downstream, before the
            # upstream inf-clamp safety net in T5Block.forward got a chance to catch it --
            # see the WORKING_DTYPE comment above), since fixed by switching to bf16.
            # `generate()` (a different code path, used for the actual predicted SQL) is a
            # separate call and is not blocked by this either way.
            print("WARNING: eval_loss is NaN; this does not affect the generated SQL from "
                  "generate(), only the reported eval_loss metric.")
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