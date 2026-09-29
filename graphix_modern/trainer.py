"""Trainer on transformers 4.57 with the legacy (4.17) training dynamics (step A6.2c).

Ports seq2seq/utils/trainer.py (Seq2SeqTrainer: evaluate/predict/prediction_step with the
project's metric flow) and seq2seq/utils/spider.py (SpiderTrainer), and replaces 4.57's
training loop by a port of 4.17's, because the two differ in training dynamics
(PORTABILIDADE.md, A6.2c; confirmed empirically by scripts/port_trace_epochs.py):
  - updates per epoch: 4.17 floor(N/GA); 4.57 ceil(N/GA) (an extra partial update);
  - micro-batches left over at the end of an epoch: in 4.17 their gradients stay in .grad
    and join the first update of the next epoch (each still / GA); in the last epoch the
    loop stops at max_steps before reaching them. 4.57 steps them alone, / r;
  - max_steps and so the LR schedule follow from the above;
  - example order: 4.17 draws one number from the global RNG to seed a torch.Generator
    and RandomSampler takes one randperm per epoch from it; 4.57 goes through accelerate's
    seedable sampler. The legacy mechanism is kept (GRAPHIX_SAMPLER_SEED overrides the
    drawn seed, for tests that need both stacks on the same order);
  - loss / gradient_accumulation_steps always (4.57 skips the division when forward()
    takes **kwargs, as the wrappers' does).
Also: a CUDA OOM in a training step is fatal (the legacy step silently skipped the
micro-batch; GRAPHIX_ALLOW_OOM_SKIP=1 restores that), and optimizer/scheduler states of
this project's own checkpoints are loaded with torch.load(weights_only=True), which
4.57 refuses under torch < 2.6 (CVE-2025-32434; see _load_optimizer_and_scheduler).
Out of scope for the study and not ported: deepspeed, TPU, apex, grad scaling, DDP.
"""
import collections
import logging
import math
import os
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer
from transformers.trainer_callback import ExportableState, TrainerState
from transformers.trainer_utils import PredictionOutput, TrainOutput, has_length, speed_metrics

from .profiling import section

logger = logging.getLogger(__name__)


def _timed_batches(dataloader):
    """Iterate a DataLoader unchanged, timing the wait for each batch (GRAPHIX_PROFILE=1)."""
    it = iter(dataloader)
    while True:
        with section("data_wait"):
            try:
                batch = next(it)
            except StopIteration:
                return
        yield batch


class LegacyRandomSampler(torch.utils.data.Sampler):
    """torch 1.9's RandomSampler.__iter__ (replacement=False): one randperm per epoch.
    torch 2.x's version also evaluates randperm(n)[:num_samples % n] after the epoch's
    permutation even when that slice is empty, drawing a second, discarded permutation
    from the generator every epoch, which changes the example order from epoch 2 on
    (seen in scripts/port_trace_epochs.py). randperm itself is identical across versions."""

    def __init__(self, data_source, generator):
        self.data_source, self.generator = data_source, generator

    def __iter__(self):
        yield from torch.randperm(len(self.data_source), generator=self.generator).tolist()

    def __len__(self):
        return len(self.data_source)


class EvalPrediction(NamedTuple):
    predictions: List[str]
    label_ids: np.ndarray
    metas: List[dict]


class LegacyLoopTrainer(Seq2SeqTrainer):
    """transformers 4.57 Seq2SeqTrainer running transformers 4.17's training loop."""

    # ---- data order: 4.17's sampler, plain DataLoader (no accelerate re-sampling) ----
    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if not has_length(dataset):
            return None
        seed = os.environ.get("GRAPHIX_SAMPLER_SEED")
        generator = torch.Generator()
        generator.manual_seed(int(seed) if seed is not None else int(torch.empty((), dtype=torch.int64).random_().item()))
        self._sampler_seed = generator.initial_seed()
        return LegacyRandomSampler(dataset, generator=generator)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return DataLoader(self.train_dataset, batch_size=self._train_batch_size,
                          sampler=self._get_train_sampler(), collate_fn=self.data_collator,
                          drop_last=self.args.dataloader_drop_last, num_workers=self.args.dataloader_num_workers,
                          pin_memory=self.args.dataloader_pin_memory)

    # ---- one micro-batch: loss / GA, plain backward, OOM fatal --------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        with section("prepare_inputs"):
            inputs = self._prepare_inputs(inputs)
        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            with section("backward"):
                loss.backward()
            return loss.detach()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or os.environ.get("GRAPHIX_ALLOW_OOM_SKIP") != "1":
                raise
            logger.warning("training_step: CUDA OOM on graph_idx=%s, micro-batch skipped (GRAPHIX_ALLOW_OOM_SKIP=1): %s",
                           inputs.get("graph_idx"), e)
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device=self.args.device)

    # ---- optimizer/scheduler state of our own checkpoints under torch < 2.6 -------------
    def _load_optimizer_and_scheduler(self, checkpoint):
        """4.57 gates these torch.load calls on torch >= 2.6 (CVE-2025-32434:
        weights_only=True can be bypassed by a crafted file under torch < 2.6). MODERN_A is
        on torch 2.4 (DGL); resuming only ever reads checkpoints this project's own runs
        wrote, never third-party files, so they are loaded with weights_only=True here."""
        if checkpoint is None:
            return
        opt, sch = os.path.join(checkpoint, "optimizer.pt"), os.path.join(checkpoint, "scheduler.pt")
        if os.path.isfile(opt) and os.path.isfile(sch):
            self.optimizer.load_state_dict(torch.load(opt, map_location=self.args.device, weights_only=True))
            self.lr_scheduler.load_state_dict(torch.load(sch, weights_only=True))

    # ---- the 4.17 training loop -----------------------------------------------------------
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None,
                             ignore_keys_for_eval=None):
        self._train_batch_size = batch_size
        args = args if args is not None else self.args
        train_dataloader = self.get_train_dataloader()
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        num_update_steps_per_epoch = max(len(train_dataloader) // args.gradient_accumulation_steps, 1)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0)
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = len(self.train_dataset) * args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState(stateful_callbacks=[
            cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)])
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size
        self.state.compute_steps(args, max_steps)

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
        model = self.model  # single GPU: no wrapping
        self.model_wrapped = model
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained, steps_trained_in_current_epoch = 0, 0
        state_file = os.path.join(resume_from_checkpoint, "trainer_state.json") if resume_from_checkpoint else None
        if state_file and os.path.isfile(state_file):
            self.state = TrainerState.load_from_json(state_file)
            self.state.compute_steps(args, max_steps)
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = (self.state.global_step % num_update_steps_per_epoch) * \
                    args.gradient_accumulation_steps
            logger.info("Resuming from global step %d (epoch %d, skipping %d micro-batches)",
                        self.state.global_step, epochs_trained, steps_trained_in_current_epoch)

        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader
        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        grad_norm = None
        model.zero_grad()
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if not args.ignore_data_skip:
            for _ in range(epochs_trained):  # advance the sampler's generator as 4.17 does
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            steps_in_epoch = len(train_dataloader)
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            step = -1
            for step, inputs in enumerate(_timed_batches(train_dataloader)):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                tr_loss_step = self.training_step(model, inputs)
                if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step
                self.current_flos += float(self.floating_point_ops(inputs))

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch):
                    with section("optimizer"):
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval,
                                                  start_time)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)
            if self.control.should_training_stop:
                break

        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / max(self.state.global_step, 1)
        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        return TrainOutput(self.state.global_step, train_loss, metrics)


class GraphixSeq2SeqTrainer(LegacyLoopTrainer):
    """Port of seq2seq/utils/trainer.py::Seq2SeqTrainer (evaluate/predict with the
    project's metric flow; prediction_step with its gen_kwargs)."""

    def __init__(self, metric, *args, eval_examples=None, ignore_pad_token_for_loss=True,
                 target_with_db_id=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.target_with_db_id = target_with_db_id
        self._max_length = self._max_time = self._num_beams = None

    def _compute_metrics(self, eval_prediction):
        raise NotImplementedError()

    def _post_process_function(self, examples, features, predictions, stage):
        raise NotImplementedError()

    def evaluate(self, *args, **kwargs):
        with section("evaluate"):
            return self._evaluate_impl(*args, **kwargs)

    def _evaluate_impl(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix="eval",
                       max_length=None, max_time=None, num_beams=None):
        self._max_length, self._max_time, self._num_beams = max_length, max_time, num_beams
        self._memory_tracker.start()
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(eval_dataloader, description="Evaluation",
                                          prediction_loss_only=True if compute_metrics is None else None,
                                          ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        finally:
            self.compute_metrics = compute_metrics
        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(eval_examples, eval_dataset, output.predictions,
                                                     "eval_{}".format(self.state.epoch))
            output.metrics.update(self.compute_metrics(eval_preds))
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(eval_dataset)))
        for key in list(output.metrics.keys()):
            if not key.startswith(metric_key_prefix + "_"):
                output.metrics[metric_key_prefix + "_" + key] = output.metrics.pop(key)
        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None, metric_key_prefix="eval",
                max_length=None, max_time=None, num_beams=None):
        self._max_length, self._max_time, self._num_beams = max_length, max_time, num_beams
        self._memory_tracker.start()
        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(test_dataloader, description="Prediction", ignore_keys=ignore_keys,
                                          metric_key_prefix=metric_key_prefix)
        finally:
            self.compute_metrics = compute_metrics
        if self.compute_metrics is not None:
            eval_preds = self._post_process_function(test_examples, test_dataset, output.predictions, metric_key_prefix)
            output.metrics.update(self.compute_metrics(eval_preds))
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))
        for key in list(output.metrics.keys()):
            if not key.startswith(metric_key_prefix + "_"):
                output.metrics[metric_key_prefix + "_" + key] = output.metrics.pop(key)
        self.log(output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only=prediction_loss_only,
                                           ignore_keys=ignore_keys)
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": False,
            "no_repeat_ngram_size": 0,  # hard-coded in the legacy trainer
        }
        if self._max_time is not None:
            gen_kwargs["max_time"] = self._max_time
        for k in ("description_input_ids", "description_attention_mask", "knowledge_input_ids",
                  "knowledge_attention_mask", "task_ids", "graph_idx"):
            if k in inputs:
                gen_kwargs[k] = inputs[k]
        generated_tokens = self.model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                               **gen_kwargs)
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None
        if self.args.prediction_loss_only:
            return (loss, None, None)
        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        return (loss, generated_tokens, labels)


class SpiderTrainer(GraphixSeq2SeqTrainer):
    """Port of seq2seq/utils/spider.py::SpiderTrainer."""

    # transformers 4.17's decode defaulted to clean_up_tokenization_spaces=True ("a, b");
    # 4.57's T5 tokenizer defaults to False ("a , b"). Fixed explicitly to the legacy value
    # so decoded inputs, labels and predicted SQL do not depend on the library default.
    CLEANUP = True

    def _post_process_function(self, examples, features, predictions, stage):
        import json
        tok = self.processing_class
        inputs = tok.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=self.CLEANUP)
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            _label_ids = np.where(label_ids != -100, label_ids, tok.pad_token_id)
        decoded_label_ids = tok.batch_decode(_label_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=self.CLEANUP)
        metas = [
            {"query": x["query"], "question": x["question"], "context": context, "label": label,
             "db_id": x["db_id"], "db_path": x["db_path"], "db_table_names": x["db_table_names"],
             "db_column_names": x["db_column_names"], "db_foreign_keys": x["db_foreign_keys"]}
            for x, context, label in zip(examples, inputs, decoded_label_ids)
        ]
        predictions = tok.batch_decode(predictions, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=self.CLEANUP)
        assert len(metas) == len(predictions)
        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
            json.dump([dict(**{"prediction": p}, **m) for p, m in zip(predictions, metas)], f, indent=4)
        return EvalPrediction(predictions=predictions, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction):
        predictions, label_ids, metas = eval_prediction
        if self.target_with_db_id:
            predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
        return self.metric.compute(predictions=predictions, references=metas)
