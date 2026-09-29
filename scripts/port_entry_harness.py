#!/usr/bin/env python
"""Run a real training entrypoint under test instrumentation (steps A6.3/A7: T7 and T10).
Python 3.7 compatible; the same script in both images.

  python scripts/port_entry_harness.py legacy|modern CONFIG.json
runs seq2seq/run_seq2seq_train.py (legacy) or graphix_modern/train.py (modern) unchanged,
after these test-only patches, each enabled by an environment variable:
  HARNESS_NO_DROPOUT=1    torch.nn.functional.dropout -> input * 1.0 (a fresh tensor, which
                          the in-place graph_caption write needs; see T6). Covers nn.Dropout
                          modules and T5Attention's functional dropout in both stacks.
  HARNESS_ORDER_LOG=path  records the dataset index of every training item fetched, in order
                          (TokenizedDataset.__getitem__ while the model is training).
  HARNESS_STOP_AT_STEP=n  (modern only) stops training after optimizer step n and saves a
                          checkpoint there, to simulate an interruption for the resume test.
Nothing else is changed; per-step loss/LR come from trainer_state.json (logging_steps=1).
"""
import json
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
env, config = sys.argv[1], sys.argv[2]

if os.environ.get("HARNESS_NO_DROPOUT") == "1":
    import torch.nn.functional as F

    def _no_dropout(input, p=0.5, training=True, inplace=False):
        return input * 1.0
    F.dropout = _no_dropout

order_log = os.environ.get("HARNESS_ORDER_LOG")
if order_log:
    from seq2seq.utils import dataset_graph
    _orig_getitem = dataset_graph.TokenizedDataset.__getitem__
    _order = []

    def _getitem(self, index):
        if getattr(self, "_harness_train", False):
            _order.append(int(index))
        return _orig_getitem(self, index)
    dataset_graph.TokenizedDataset.__getitem__ = _getitem
    _orig_init = dataset_graph.TokenizedDataset.__init__

    def _init(self, data_training_args, training_args, tokenizer, seq2seq_dataset=None, graph_pedia=None):
        _orig_init(self, data_training_args, training_args, tokenizer, seq2seq_dataset, graph_pedia)
        # the first TokenizedDataset built by the entrypoints is the training one
        if not getattr(dataset_graph, "_harness_seen_train", False):
            self._harness_train = True
            dataset_graph._harness_seen_train = True
    dataset_graph.TokenizedDataset.__init__ = _init

    import atexit
    atexit.register(lambda: json.dump(_order, open(order_log, "w")))

stop_at = os.environ.get("HARNESS_STOP_AT_STEP")
if stop_at and env == "modern":
    from transformers import TrainerCallback
    from graphix_modern import trainer as T

    class _StopAt(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= int(stop_at):
                control.should_save = True
                control.should_training_stop = True

    _orig = T.LegacyLoopTrainer.__init__

    def _tinit(self, *a, **k):
        _orig(self, *a, **k)
        self.add_callback(_StopAt())
    T.LegacyLoopTrainer.__init__ = _tinit

sys.argv = ["entry", config]
if env == "legacy":
    sys.path.insert(0, "seq2seq")
    runpy.run_path("seq2seq/run_seq2seq_train.py", run_name="__main__")
else:
    runpy.run_path("graphix_modern/train.py", run_name="__main__")
