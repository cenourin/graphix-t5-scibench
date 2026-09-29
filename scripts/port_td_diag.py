#!/usr/bin/env python
"""Diagnostic for TD's P check: decode the HF-preprocessed dev inputs/labels of given
examples, in one environment, so the two environments' texts can be diffed.
  python scripts/port_td_diag.py ENV BENCH IDX [IDX ...]
Writes data_all_in/data/port_tests/TD/diag_<ENV>_<BENCH>.json. Python 3.7 compatible.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from port_td_data import BENCH, OUT, build_args  # noqa: E402


def main(env, bench, idxs):
    import rapidfuzz
    from port_t1_tokenizer import load_tokenizer
    model_args, data_args, dta, targs = build_args(BENCH[bench]["config"])
    targs.do_train = False  # only the dev split is needed here
    tok = load_tokenizer()
    if env == "legacy":
        from seq2seq.utils.dataset_loader import load_dataset
        metric, splits = load_dataset(data_args=data_args, model_args=model_args, data_training_args=dta,
                                      training_args=targs, tokenizer=tok)
    else:
        from graphix_modern.data import load_dataset_splits
        metric, splits = load_dataset_splits(data_args, model_args, dta, targs, tok)
    ev = splits.eval_split.dataset
    out = {"rapidfuzz": rapidfuzz.__version__, "examples": {}}
    for i in idxs:
        # raw ids, plus a decode with the cleanup made explicit (its default differs by version)
        out["examples"][i] = {"input_ids": list(ev[i]["input_ids"]), "labels": list(ev[i]["labels"]),
                              "input": tok.decode(ev[i]["input_ids"], clean_up_tokenization_spaces=False),
                              "labels_text": tok.decode(ev[i]["labels"], clean_up_tokenization_spaces=False)}
    json.dump(out, open(str(OUT / ("diag_%s_%s.json" % (env, bench))), "w"), indent=1)
    print(env, bench, "rapidfuzz", rapidfuzz.__version__, "wrote", len(idxs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])
