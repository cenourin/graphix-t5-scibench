#!/usr/bin/env python
"""A4 prerequisite (PORTABILIDADE.md §4): the stock t5-base checkpoint as safetensors.

transformers 4.57.6 refuses to torch.load a pickled pytorch_model.bin under torch < 2.6
(CVE-2025-32434), and MODERN_A must stay on torch 2.4.0 (dgl 2.4.0 requires torch<=2.4.0).
So the stock checkpoint is converted once, pickle-free, to data_all_in/t5-base-st/, and
checked tensor by tensor, bit for bit, against the original. data_all_in/t5-base/ (used
by the legacy image) is left untouched.
Run in the graphix-modern image: python scripts/port_convert_t5_safetensors.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, ".")
SRC, DST = Path("data_all_in/t5-base"), Path("data_all_in/t5-base-st")
COPY = ["config.json", "spiece.model", "tokenizer_config.json", "special_tokens_map.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    DST.mkdir(parents=True, exist_ok=True)
    # weights_only=True: tensors only, no arbitrary pickle objects
    sd = torch.load(str(SRC / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    # safetensors refuses tensors sharing storage (T5 ties embeddings): store independent copies
    save_file({k: v.detach().clone().contiguous() for k, v in sd.items()}, str(DST / "model.safetensors"),
              metadata={"format": "pt", "source": "data_all_in/t5-base/pytorch_model.bin"})
    for f in COPY:
        shutil.copy2(str(SRC / f), str(DST / f))
    back = load_file(str(DST / "model.safetensors"))
    assert sorted(back) == sorted(sd), "key sets differ"
    bad = [k for k in sd if sd[k].dtype != back[k].dtype or sd[k].shape != back[k].shape or not torch.equal(sd[k], back[k])]
    assert not bad, bad
    manifest = {"source": {"pytorch_model.bin": sha256(SRC / "pytorch_model.bin")},
                "files": {p.name: sha256(p) for p in sorted(DST.iterdir()) if p.name != "manifest.json"},
                "tensors": len(sd), "check": "all tensors equal bit for bit (dtype, shape, values)"}
    json.dump(manifest, open(DST / "manifest.json", "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
