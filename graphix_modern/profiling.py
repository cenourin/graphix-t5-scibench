"""Per-stage timing for the modern stack (step A6.2e; PORTABILIDADE.md, "Profiling desde a
Fase A"). Off unless GRAPHIX_PROFILE=1, and then it only measures: no change of order,
dtype or device. When on, every section boundary calls torch.cuda.synchronize() so GPU
time lands in the right section; that slows the run, so throughput numbers come from runs
with profiling off.

Sections (nested ones are reported both as measured and, for the enclosing ones, net of
what they contain):
  data_wait       waiting for the next training batch (DataLoader + collate)
  prepare_inputs  CPU -> GPU copy of the batch (Trainer._prepare_inputs)
  graph_build     graph_batch assembly from the graph store (wrapper.graph_factory)
  graph_to        graph.to(device) inside each RGAT layer
  rgat            each T5LayerRGAT forward (includes graph_to)
  encoder         the T5 encoder stack forward (includes rgat)
  decoder         the T5 decoder stack forward
  lm_head         the output projection
  backward        loss.backward()
  optimizer       optimizer.step() + lr_scheduler.step() + zero_grad()
  evaluate        a whole evaluation pass
"""
import contextlib
import json
import os
import time
from collections import defaultdict

ENABLED = os.environ.get("GRAPHIX_PROFILE") == "1"
_totals = defaultdict(float)
_counts = defaultdict(int)


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def section(name):
    if not ENABLED:
        yield
        return
    _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        _totals[name] += time.perf_counter() - t0
        _counts[name] += 1


def attach_module_hooks(module, name):
    """Time a module's forward as `name` through forward pre/post hooks."""
    if not ENABLED:
        return []
    starts = []

    def pre(mod, inputs):
        _sync()
        starts.append(time.perf_counter())

    def post(mod, inputs, output):
        _sync()
        _totals[name] += time.perf_counter() - starts.pop()
        _counts[name] += 1

    return [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]


def attach_model_hooks(model):
    """Hooks on a wrapper (RGATModel/PlainModel): encoder, decoder, lm_head, every RGAT layer."""
    if not ENABLED:
        return []
    t5 = model.pretrain_model
    handles = attach_module_hooks(t5.encoder, "encoder") + attach_module_hooks(t5.decoder, "decoder") \
        + attach_module_hooks(t5.lm_head, "lm_head")
    for block in t5.encoder.block:
        if hasattr(block, "rgat_layer"):
            handles += attach_module_hooks(block.rgat_layer, "rgat")
    return handles


def summary():
    s = {k: {"seconds": round(v, 4), "calls": _counts[k]} for k, v in sorted(_totals.items())}
    if "encoder" in _totals:
        s["encoder_net_of_rgat"] = {"seconds": round(_totals["encoder"] - _totals.get("rgat", 0.0), 4)}
    if "rgat" in _totals:
        s["rgat_net_of_graph_to"] = {"seconds": round(_totals["rgat"] - _totals.get("graph_to", 0.0), 4)}
    return s


def dump(output_dir):
    if not ENABLED:
        return None
    s = summary()
    with open(os.path.join(output_dir, "profile.json"), "w") as f:
        json.dump(s, f, indent=1)
    return s
