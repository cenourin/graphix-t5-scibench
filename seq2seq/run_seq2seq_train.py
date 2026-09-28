# Set up logging
# We use Picard as our codebase, for more details please refer to https://github.com/ServiceNow/picard
import sys
import logging
import pdb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

import os
import json
import re
import shelve
from pathlib import Path
import pickle
import torch
from contextlib import nullcontext
from dataclasses import asdict, fields
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.models.auto import AutoConfig, AutoTokenizer
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint, set_seed
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tokenizers import AddedToken
from seq2seq.utils.args import ModelArguments
from seq2seq.utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
from seq2seq.utils.dataset import DataTrainingArguments, DataArguments
from seq2seq.utils.dataset_loader import load_dataset
from seq2seq.utils.spider import SpiderTrainer
from seq2seq.utils.cosql import CoSQLTrainer
from seq2seq.utils.dataset_graph import TokenizedDataset, get_graph_entry

print(os.getcwd())


def _load_graph_pedia(path):
    # ScienceBenchmark's train graph_pedia is written by
    # data_all_in/run_train_subword_and_graph.py as an incremental shelve store
    # (dbm.dumb backend -> path + ".dat"/".dir"/".bak" on disk), not a single pickled
    # dict -- see that script's module docstring. Detect which format `path` is and
    # open accordingly; Spider's graph_pedia_total.bin (plain pickle) is unaffected.
    if os.path.exists(path + '.dat'):
        return shelve.open(path, flag='r')
    return pickle.load(open(path, 'rb'))


# Separate train/eval graph_pedia paths (both default to Spider's single merged file,
# preserving old behavior exactly). This split matters for ScienceBenchmark: unlike
# Spider (whose graph_pedia_total.bin merges train+dev with dev's graph_idx offset by
# 8577 to avoid collisions -- see inject_syntax.py), ScienceBenchmark's dev graph_idx
# is 0-based (--dev_graph_idx_offset 0), the same range as train's, so train and dev
# examples cannot share one graph_pedia dict/store without index collisions -- they
# have to stay in the two separate files run_train_subword_and_graph.py / the normal
# dev pipeline each produced.
graph_pedia_train = _load_graph_pedia(
    os.environ.get('GRAPHIX_TRAIN_GRAPH_PEDIA_PATH', 'data_all_in/data/output/graph_pedia_total.bin'))
graph_pedia_eval = _load_graph_pedia(
    os.environ.get('GRAPHIX_EVAL_GRAPH_PEDIA_PATH', 'data_all_in/data/output/graph_pedia_total.bin'))
seq2seq_train_dataset = json.load(open(
    os.environ.get('GRAPHIX_TRAIN_DATASET_PATH', 'data_all_in/data/output/seq2seq_train_dataset.json'), 'r'))
seq2seq_eval_dataset = json.load(open(
    os.environ.get('GRAPHIX_EVAL_DATASET_PATH', 'data_all_in/data/output/seq2seq_dev_dataset.json'), 'r'))

# On 8GB VRAM, examples whose RGAT graph exceeds MAX_GRAPH_NODES nodes OOM during
# training (the DGL propagate_attention pass, not the T5 encoder/decoder itself --
# confirmed via a pilot run that failed identically with and without
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments, ruling out fragmentation). Dropping
# these examples (the top ~3% by graph size on Spider train) is a hardware
# accommodation, not a change to how any *retained* example is processed --
# TokenizedDataset still tokenizes/serializes every kept example exactly as before.
MAX_GRAPH_NODES = int(os.environ.get("GRAPHIX_MAX_GRAPH_NODES", "512"))


def _filter_by_graph_size(dataset, graph_pedia, max_nodes):
    kept = [
        item for item in dataset
        if get_graph_entry(graph_pedia, item['graph_idx'])['graph'].number_of_nodes() <= max_nodes
    ]
    dropped = len(dataset) - len(kept)
    if dropped:
        logger.warning(
            "Dropped %d/%d examples with RGAT graph > %d nodes (VRAM accommodation)",
            dropped, len(dataset), max_nodes,
        )
    return kept


# Remember each eval example's position in the unfiltered file: the metric pairs the
# i-th prediction with the i-th HF-loaded example (dataset_splits.eval_split.examples),
# which is never filtered, so main() must drop the same positions there too.
for _pos, _item in enumerate(seq2seq_eval_dataset):
    _item['_eval_pos'] = _pos

seq2seq_train_dataset = _filter_by_graph_size(seq2seq_train_dataset, graph_pedia_train, MAX_GRAPH_NODES)
seq2seq_eval_dataset = _filter_by_graph_size(seq2seq_eval_dataset, graph_pedia_eval, MAX_GRAPH_NODES)


# TokenizedDataset.__getitem__ (dataset_graph.py) only *prints* when its tokenized input
# length doesn't match the RGAT graph's node count -- the real assert is commented out
# there, by design. That mismatch crashes deep in the RGAT/DGL forward pass (feature
# count != node count) instead of failing cleanly here. Needs the tokenizer, so this
# filter (unlike _filter_by_graph_size above) has to run inside main(), after the
# tokenizer is loaded -- see the call site there.
def _filter_by_token_node_match(dataset, graph_pedia, tokenizer, max_source_length):
    def _match(raw_item):
        question_in = " ".join(raw_item['raw_question_toks'])
        struct_in_norm = re.sub('  +', ' ', get_graph_entry(graph_pedia, raw_item['graph_idx'])['new_struct_in'])
        seq_in = "{} ; {}".format(question_in, struct_in_norm)
        tokenized = tokenizer(seq_in, max_length=max_source_length, truncation=True)
        n_tokens = len([a for a in tokenized.input_ids if a > 1])
        n_nodes = get_graph_entry(graph_pedia, raw_item['graph_idx'])['graph'].number_of_nodes()
        return n_tokens == n_nodes

    matches = [_match(item) for item in dataset]
    kept = [item for item, ok in zip(dataset, matches) if ok]
    dropped = len(dataset) - len(kept)
    if dropped:
        from collections import Counter
        dropped_by_db = Counter(item.get('db_id') for item, ok in zip(dataset, matches) if not ok)
        logger.warning(
            "Dropped %d/%d examples with a token/graph-node count mismatch (by db_id: %s)",
            dropped, len(dataset), dict(dropped_by_db),
        )
    return kept


def main() -> None:
    # See all possible arguments by passing the --help flag to this script.
    parser = HfArgumentParser(
        (PicardArguments, ModelArguments, DataArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    picard_args: PicardArguments
    model_args: ModelArguments
    data_args: DataArguments
    data_training_args: DataTrainingArguments
    training_args: Seq2SeqTrainingArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        picard_args, model_args, data_args, data_training_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 3 and sys.argv[1].startswith("--local_rank") and sys.argv[2].endswith(".json"):
        data = json.loads(Path(os.path.abspath(sys.argv[2])).read_text())
        data.update({"local_rank": int(sys.argv[1].split("=")[1])})
        picard_args, model_args, data_args, data_training_args, training_args = parser.parse_dict(args=data)
    else:
        picard_args, model_args, data_args, data_training_args, training_args = parser.parse_args_into_dataclasses()
    
    # If model_name_or_path includes ??? instead of the number of steps, 
    # we load the latest checkpoint.
    if 'checkpoint-???' in model_args.model_name_or_path:
        model_args.model_name_or_path = get_last_checkpoint(
            os.path.dirname(model_args.model_name_or_path))
        logger.info(f"Resolve model_name_or_path to {model_args.model_name_or_path}")

    combined_args_dict = {
        **asdict(picard_args),
        **asdict(model_args),
        **asdict(data_args),
        **asdict(data_training_args),
        **training_args.to_sanitized_dict(),
    }
    combined_args_dict.pop("local_rank", None)

    if "wandb" in training_args.report_to and training_args.local_rank <= 0:
        import wandb

        init_args = {}
        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "text-to-sql"),
            name=training_args.run_name,
            **init_args,
        )
        wandb.config.update(combined_args_dict, allow_val_change=True)

    if not training_args.do_train and not training_args.do_eval and not training_args.do_predict:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    os.makedirs(training_args.output_dir, exist_ok=True)

    if training_args.local_rank <= 0:
        with open(f"{training_args.output_dir}/combined_args.json", "w") as f:
            json.dump(combined_args_dict, f, indent=4)

    # Initialize random number generators
    set_seed(training_args.seed)

    # Initialize config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        max_length=data_training_args.max_target_length,
        num_beams=data_training_args.num_beams,
        num_beam_groups=data_training_args.num_beam_groups,
        diversity_penalty=data_training_args.diversity_penalty,
        no_repeat_ngram_size=data_training_args.no_repeat_ngram_size,
        gradient_checkpointing=training_args.gradient_checkpointing,
        use_cache=not training_args.gradient_checkpointing,
        # use_cache=False
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Only fast tokenizers are currently supported"
    if isinstance(tokenizer, T5TokenizerFast):
        # In T5 `<` is OOV, see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/restore_oov.py
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    global seq2seq_train_dataset, seq2seq_eval_dataset
    seq2seq_train_dataset = _filter_by_token_node_match(
        seq2seq_train_dataset, graph_pedia_train, tokenizer, data_training_args.max_source_length)
    seq2seq_eval_dataset = _filter_by_token_node_match(
        seq2seq_eval_dataset, graph_pedia_eval, tokenizer, data_training_args.max_source_length)

    # Load dataset
    metric, dataset_splits = load_dataset(
        data_args=data_args,
        model_args=model_args,
        data_training_args=data_training_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )

    # Without this, any filtered eval example shifted every later prediction onto the
    # wrong gold SQL (ScienceBenchmark dev: 6 dropped from position 216 on, 76/293
    # predictions scored against another example's gold in all runs before this fix).
    # Only when metrics are computed: with GRAPHIX_LOSS_ONLY_EVAL the eval file may be a
    # validation slice carved from train, which has no counterpart in the HF dev split.
    eval_split = dataset_splits.eval_split
    scored_eval = eval_split is not None and os.environ.get("GRAPHIX_LOSS_ONLY_EVAL") != "1"
    if scored_eval and data_training_args.max_val_samples is not None:
        # eval_split.examples is truncated to the first max_val_samples; match it.
        seq2seq_eval_dataset = [it for it in seq2seq_eval_dataset
                                if it['_eval_pos'] < data_training_args.max_val_samples]
    if scored_eval and len(eval_split.examples) != len(seq2seq_eval_dataset):
        kept = [item['_eval_pos'] for item in seq2seq_eval_dataset]
        assert len(eval_split.examples) > max(kept), "eval dataset file and HF dev split differ"
        eval_split.examples = eval_split.examples.select(kept)
        logger.warning("Aligned eval examples to the %d filtered eval items", len(kept))

    # max_train_samples was previously honored only by the HF-datasets split and never
    # reached this TokenizedDataset, which trained on every example. No existing train
    # config sets it, so behavior is unchanged for them; it now enables tiny smoke runs.
    if data_training_args.max_train_samples is not None:
        seq2seq_train_dataset = seq2seq_train_dataset[:data_training_args.max_train_samples]
    train_dataset = TokenizedDataset(data_training_args, training_args, tokenizer,
                                     seq2seq_train_dataset, graph_pedia_train) if seq2seq_train_dataset else None
    eval_dataset = TokenizedDataset(data_training_args, training_args, tokenizer,
                                    seq2seq_eval_dataset, graph_pedia_eval) if seq2seq_eval_dataset else None
    

    # Initialize Picard if necessary
    with PicardLauncher() if picard_args.launch_picard and training_args.local_rank <= 0 else nullcontext(None):
        # Get Picard model class wrapper
        if picard_args.use_picard:
            model_cls_wrapper = lambda model_cls: with_picard(
                model_cls=model_cls, picard_args=picard_args, tokenizer=tokenizer, schemas=dataset_splits.schemas
            )
        else:
            model_cls_wrapper = lambda model_cls: model_cls

        # Initialize model
        '''We load our own model: '''
        # GRAPHIX_MODEL_VARIANT=plain selects models.graphix.plain.Model (a standard,
        # non-RGAT-injected T5ForConditionalGeneration wrapper) instead of the default
        # RGAT-aware one -- used for the "plain T5, no graph structure" ablation
        # baseline. graph_idx still flows through TokenizedDataset either way (it's
        # needed there to reconstruct the serialized-schema input text), but plain.Model
        # never builds a graph_batch or touches DGL.
        if os.environ.get("GRAPHIX_MODEL_VARIANT", "rgat") == "plain":
            from models.graphix.plain import Model
        else:
            from models.graphix.rgat import Model
        model = Model(tokenizer, model_cls_wrapper, model_args, config, graph_pedia_train, graph_pedia_eval)
        # model_name_or_path pointing at a checkpoint saved by this script does NOT load
        # its weights (keys carry a `pretrain_model.` prefix that from_pretrained ignores;
        # see RISCOS.md R0.1). To evaluate a trained run, keep model_name_or_path on the
        # stock T5 dir and point this at the run's pytorch_model.bin: strict, so any key
        # mismatch fails loudly instead of silently evaluating untrained weights.
        init_state_dict = os.environ.get("GRAPHIX_INIT_STATE_DICT")
        if init_state_dict:
            model.load_state_dict(torch.load(init_state_dict, map_location="cpu"), strict=True)
            logger.info("Loaded weights (strict) from %s", init_state_dict)
    
        if isinstance(model, T5ForConditionalGeneration):
            model.resize_token_embeddings(len(tokenizer))

        if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )
        # Initialize Trainer
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "metric": metric,
            "train_dataset": train_dataset if training_args.do_train else None,
            "eval_dataset": eval_dataset if training_args.do_eval else None,
            "eval_examples": dataset_splits.eval_split.examples if training_args.do_eval else None,
            "tokenizer": tokenizer,
            "data_collator": DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=(-100 if data_training_args.ignore_pad_token_for_loss else tokenizer.pad_token_id),
                pad_to_multiple_of=8 if training_args.fp16 else None,
            ),
            "ignore_pad_token_for_loss": data_training_args.ignore_pad_token_for_loss,
            "target_with_db_id": data_training_args.target_with_db_id,
        }
        # pdb.set_trace()
        #using spidertrainer as it is.
        if data_args.dataset in ["spider", "spider_realistic", "spider_syn", "spider_dk", "sciencebenchmark"]:
            trainer = SpiderTrainer(**trainer_kwargs)
        elif data_args.dataset in ["cosql", "cosql+spider"]:
            trainer = CoSQLTrainer(**trainer_kwargs)
        else:
            raise NotImplementedError()

        # Opt-in extras, all off unless their env var is set, so existing configs/runs
        # behave exactly as before. Early stopping needs load_best_model_at_end +
        # metric_for_best_model (already set in every train config).
        patience = os.environ.get("GRAPHIX_EARLY_STOPPING_PATIENCE")
        if patience:
            from transformers import EarlyStoppingCallback
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=int(patience)))
        # Loss-only evaluation: skip generation + exact-match/exec scoring (used by the
        # Optuna search, where only eval_loss is needed and generation dominates eval time).
        if os.environ.get("GRAPHIX_LOSS_ONLY_EVAL") == "1":
            trainer.compute_metrics = None
        # Stop after N epochs while keeping num_train_epochs (and so the LR schedule) as
        # configured: an Optuna trial is then an exact prefix of the final run it predicts.
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
                storage=os.environ["GRAPHIX_OPTUNA_STORAGE"],
                study_name=os.environ["GRAPHIX_OPTUNA_STUDY"],
                trial_id=int(os.environ["GRAPHIX_OPTUNA_TRIAL_ID"]),
                pruned_marker=os.environ["GRAPHIX_OPTUNA_PRUNED_MARKER"],
            ))

        # Training
        if training_args.do_train:
            logger.info("*** Train ***")

            checkpoint = None

            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint

            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = (
                data_training_args.max_train_samples
                if data_training_args.max_train_samples is not None
                else len(dataset_splits.train_split.dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(dataset_splits.train_split.dataset))
            if torch.cuda.is_available():
                metrics["train_peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                metrics["train_peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
                metrics["train_gpu"] = torch.cuda.get_device_name(0)

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate(
                max_length=data_training_args.val_max_target_length,
                max_time=data_training_args.val_max_time,
                num_beams=data_training_args.num_beams,
                metric_key_prefix="eval",
            )
            max_val_samples = (
                data_training_args.max_val_samples
                if data_training_args.max_val_samples is not None
                else len(dataset_splits.eval_split.dataset)
            )
            metrics["eval_samples"] = min(max_val_samples, len(dataset_splits.eval_split.dataset))
            if torch.cuda.is_available():
                metrics["eval_peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                metrics["eval_peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
                metrics["eval_gpu"] = torch.cuda.get_device_name(0)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # Testing
        if training_args.do_predict:
            logger.info("*** Predict ***")
            for section, test_split in dataset_splits.test_splits.items():
                results = trainer.predict(
                    test_split.dataset, 
                    test_split.examples,
                    max_length=data_training_args.val_max_target_length,
                    max_time=data_training_args.val_max_time,
                    num_beams=data_training_args.num_beams,
                    metric_key_prefix=section)
                metrics = results.metrics

                metrics[f"{section}_samples"] = len(test_split.dataset)

                trainer.log_metrics(section, metrics)
                trainer.save_metrics(section, metrics)


if __name__ == "__main__":
    main()
