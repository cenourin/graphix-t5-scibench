"""Training/evaluation entrypoint on the modern stack (step A6.2d): port of
seq2seq/run_seq2seq_train.py with the same interface (one JSON config argument, the same
GRAPHIX_* environment variables), so the study orchestrator and the Optuna search can switch
entrypoints without other changes.

  python graphix_modern/train.py CONFIG.json

Differences from the legacy entrypoint, all interface-level:
  - graph_pedia comes from the A0 export (graphix_modern.data.GraphStore); the
    GRAPHIX_*_GRAPH_PEDIA_PATH variables may still hold the legacy .bin paths;
  - config key "evaluation_strategy" is read as "eval_strategy" (renamed in transformers);
  - PICARD is not supported (use_picard must be false; the study never used it);
  - GRAPHIX_INIT_STATE_DICT accepts .safetensors, or a .bin/.pt state_dict loaded with
    weights_only=True (this project's own checkpoints only, see graphix_modern/trainer.py);
  - the trained model is saved as safetensors (transformers 4.57's default).
Everything else (seeds, filters, eval alignment, early stopping, loss-only eval, Optuna
pruning, stop-after-epochs, TF32 guard, the CUDA guard for the RGAT arm, peak VRAM metrics)
follows the legacy entrypoint line by line.
"""
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from tokenizers import AddedToken  # noqa: E402
from transformers import (AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq, HfArgumentParser,  # noqa: E402
                          PreTrainedTokenizerFast, Seq2SeqTrainingArguments)
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint, set_seed  # noqa: E402

from graphix_modern import data as D  # noqa: E402
from seq2seq.utils.args import ModelArguments  # noqa: E402
from seq2seq.utils.dataset import DataArguments, DataTrainingArguments  # noqa: E402
from seq2seq.utils.dataset_graph import TokenizedDataset  # noqa: E402

logger = logging.getLogger(__name__)

SPIDER_OUT = "data_all_in/data/output"


@dataclass
class PicardArguments:
    """Accepts the legacy configs' PICARD keys; PICARD itself is not supported here."""
    use_picard: bool = field(default=True)
    launch_picard: bool = field(default=True)
    picard_host: str = field(default="localhost")
    picard_port: int = field(default=9090)
    picard_mode: str = field(default="parse_with_guards")
    picard_schedule: str = field(default="incremental")
    picard_max_tokens_to_check: int = field(default=2)


def parse_config(path):
    cfg = json.load(open(path))
    if "evaluation_strategy" in cfg:  # renamed in transformers 4.46
        cfg["eval_strategy"] = cfg.pop("evaluation_strategy")
    parser = HfArgumentParser((PicardArguments, ModelArguments, DataArguments, DataTrainingArguments,
                               Seq2SeqTrainingArguments))
    # transformers 4.17's parse_json_file silently ignored keys no dataclass declares (the
    # study configs carry one: "adam_eps", never read -- TrainingArguments' field is
    # adam_epsilon, and Adafactor does not use it). 4.57 refuses them; keep the legacy
    # behavior but say which keys are ignored.
    from dataclasses import fields
    known = {f.name for dc in parser.dataclass_types for f in fields(dc)}
    ignored = sorted(k for k in cfg if k not in known)
    if ignored:
        logger.warning("Config keys ignored (no argument declares them, as in the legacy entrypoint): %s", ignored)
    return parser.parse_dict(cfg, allow_extra_keys=True)


def load_init_state_dict(model, path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
        # safetensors files store a tied weight once (under shared.weight); the keys tied
        # to it are restored from it, and only those, before the strict load.
        shared = "pretrain_model.shared.weight"
        for key in getattr(model, "_tied_weights_keys", []):
            if key not in state and shared in state:
                state[key] = state[shared]
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    logger.info("Loaded weights (strict) from %s", path)


def main():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
    picard_args, model_args, data_args, dta, targs = parse_config(os.path.abspath(sys.argv[1]))
    if picard_args.use_picard:
        raise SystemExit("PICARD is not supported on the modern stack (set use_picard: false)")
    combined = {**asdict(picard_args), **asdict(model_args), **asdict(data_args), **asdict(dta),
                **targs.to_sanitized_dict()}
    combined.pop("local_rank", None)
    if not targs.do_train and not targs.do_eval and not targs.do_predict:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    last_checkpoint = None
    if os.path.isdir(targs.output_dir) and targs.do_train and not targs.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(targs.output_dir)
        if last_checkpoint is None and len(os.listdir(targs.output_dir)) > 0:
            raise ValueError("Output directory (%s) already exists and is not empty. "
                             "Use --overwrite_output_dir to overcome." % targs.output_dir)
    os.makedirs(targs.output_dir, exist_ok=True)
    with open(os.path.join(targs.output_dir, "combined_args.json"), "w") as f:
        json.dump(combined, f, indent=4, default=str)

    # --- data (legacy: module level + main) ---------------------------------------------
    train_store_path = os.environ.get("GRAPHIX_TRAIN_GRAPH_PEDIA_PATH", SPIDER_OUT + "/graph_pedia_total.bin")
    eval_store_path = os.environ.get("GRAPHIX_EVAL_GRAPH_PEDIA_PATH", SPIDER_OUT + "/graph_pedia_total.bin")
    store_train = D.GraphStore(D.resolve_export(train_store_path))
    store_eval = store_train if eval_store_path == train_store_path else D.GraphStore(D.resolve_export(eval_store_path))
    train_data = json.load(open(os.environ.get("GRAPHIX_TRAIN_DATASET_PATH", SPIDER_OUT + "/seq2seq_train_dataset.json")))
    eval_data = D.tag_eval_positions(json.load(open(os.environ.get("GRAPHIX_EVAL_DATASET_PATH",
                                                                   SPIDER_OUT + "/seq2seq_dev_dataset.json"))))
    max_nodes = int(os.environ.get("GRAPHIX_MAX_GRAPH_NODES", "512"))
    train_data = D.filter_by_graph_size(train_data, store_train, max_nodes)
    eval_data = D.filter_by_graph_size(eval_data, store_eval, max_nodes)

    set_seed(targs.seed)
    tf32 = os.environ.get("GRAPHIX_ALLOW_TF32") == "1"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, revision=model_args.model_revision,
        max_length=dta.max_target_length, num_beams=dta.num_beams, num_beam_groups=dta.num_beam_groups,
        diversity_penalty=dta.diversity_penalty, no_repeat_ngram_size=dta.no_repeat_ngram_size,
        gradient_checkpointing=targs.gradient_checkpointing, use_cache=not targs.gradient_checkpointing)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer, revision=model_args.model_revision)
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Only fast tokenizers are currently supported"
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    train_data = D.filter_by_token_node_match(train_data, store_train, tokenizer, dta.max_source_length)
    eval_data = D.filter_by_token_node_match(eval_data, store_eval, tokenizer, dta.max_source_length)
    metric, splits = D.load_dataset_splits(data_args, model_args, dta, targs, tokenizer)
    scored_eval = splits.eval_split is not None and os.environ.get("GRAPHIX_LOSS_ONLY_EVAL") != "1"
    eval_data = D.align_eval_examples(splits.eval_split, eval_data, dta.max_val_samples, scored_eval)
    if dta.max_train_samples is not None:
        train_data = train_data[:dta.max_train_samples]
    train_dataset = TokenizedDataset(dta, targs, tokenizer, train_data, store_train) if train_data else None
    eval_dataset = TokenizedDataset(dta, targs, tokenizer, eval_data, store_eval) if eval_data else None

    # --- model --------------------------------------------------------------------------
    from graphix_modern.models import PlainModel, RGATModel
    if os.environ.get("GRAPHIX_MODEL_VARIANT", "rgat") == "plain":
        model = PlainModel(tokenizer, model_args, config, store_train, store_eval)
    else:
        if not torch.cuda.is_available() and os.environ.get("GRAPHIX_ALLOW_CPU_RGAT") != "1":
            raise SystemExit("RGAT variant requires CUDA and none is available "
                             "(set GRAPHIX_ALLOW_CPU_RGAT=1 only for deliberate CPU tests)")
        model = RGATModel(tokenizer, model_args, config, store_train, store_eval)
    if os.environ.get("GRAPHIX_INIT_STATE_DICT"):
        load_init_state_dict(model, os.environ["GRAPHIX_INIT_STATE_DICT"])
    from graphix_modern import profiling
    profiling.attach_model_hooks(model)  # no-op unless GRAPHIX_PROFILE=1

    # --- trainer ------------------------------------------------------------------------
    from graphix_modern.trainer import SpiderTrainer
    if data_args.dataset not in ("spider", "spider_realistic", "spider_syn", "spider_dk", "sciencebenchmark"):
        raise NotImplementedError(data_args.dataset)
    trainer = SpiderTrainer(
        metric=metric, model=model, args=targs,
        train_dataset=train_dataset if targs.do_train else None,
        eval_dataset=eval_dataset if targs.do_eval else None,
        eval_examples=splits.eval_split.examples if targs.do_eval else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model,
            label_pad_token_id=(-100 if dta.ignore_pad_token_for_loss else tokenizer.pad_token_id),
            pad_to_multiple_of=8 if targs.fp16 else None),
        ignore_pad_token_for_loss=dta.ignore_pad_token_for_loss,
        target_with_db_id=dta.target_with_db_id,
    )
    patience = os.environ.get("GRAPHIX_EARLY_STOPPING_PATIENCE")
    if patience:
        from transformers import EarlyStoppingCallback
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=int(patience)))
    if os.environ.get("GRAPHIX_LOSS_ONLY_EVAL") == "1":
        trainer.compute_metrics = None
    stop_after = os.environ.get("GRAPHIX_STOP_AFTER_EPOCHS")
    if stop_after:
        from transformers import TrainerCallback

        class _StopAfterEpochs(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                if state.epoch is not None and state.epoch >= int(stop_after) - 1e-6:
                    control.should_training_stop = True

        trainer.add_callback(_StopAfterEpochs())
    if os.environ.get("GRAPHIX_OPTUNA_TRIAL_ID"):
        from seq2seq.utils.optuna_callback import OptunaPruningCallback
        trainer.add_callback(OptunaPruningCallback(
            storage=os.environ["GRAPHIX_OPTUNA_STORAGE"], study_name=os.environ["GRAPHIX_OPTUNA_STUDY"],
            trial_id=int(os.environ["GRAPHIX_OPTUNA_TRIAL_ID"]),
            pruned_marker=os.environ["GRAPHIX_OPTUNA_PRUNED_MARKER"]))

    def vram(metrics, prefix):
        if torch.cuda.is_available():
            metrics[prefix + "_peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            metrics[prefix + "_peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
            metrics[prefix + "_gpu"] = torch.cuda.get_device_name(0)

    if targs.do_train:
        checkpoint = targs.resume_from_checkpoint if targs.resume_from_checkpoint is not None else last_checkpoint
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = result.metrics
        n = dta.max_train_samples if dta.max_train_samples is not None else len(splits.train_split.dataset)
        metrics["train_samples"] = min(n, len(splits.train_split.dataset))
        vram(metrics, "train")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        profiling.dump(targs.output_dir)

    if targs.do_eval:
        metrics = trainer.evaluate(max_length=dta.val_max_target_length, max_time=dta.val_max_time,
                                   num_beams=dta.num_beams, metric_key_prefix="eval")
        n = dta.max_val_samples if dta.max_val_samples is not None else len(splits.eval_split.dataset)
        metrics["eval_samples"] = min(n, len(splits.eval_split.dataset))
        vram(metrics, "eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if targs.do_predict:
        for section, test_split in splits.test_splits.items():
            results = trainer.predict(test_split.dataset, test_split.examples, max_length=dta.val_max_target_length,
                                      max_time=dta.val_max_time, num_beams=dta.num_beams, metric_key_prefix=section)
            metrics = results.metrics
            metrics[section + "_samples"] = len(test_split.dataset)
            trainer.log_metrics(section, metrics)
            trainer.save_metrics(section, metrics)


if __name__ == "__main__":
    main()
