import os
import torch
from torch.utils.data import Dataset
import pdb
import re


def get_graph_entry(graph_pedia, graph_idx):
    # graph_pedia is either a plain dict keyed by int (existing pickled
    # graph_pedia_{dev,total}.bin files, unchanged) or a shelve store keyed by str
    # (data_all_in/run_train_subword_and_graph.py's incremental-write output, used to
    # bound preprocessing memory on ScienceBenchmark's larger schemas -- see that
    # script's module docstring). Try int first since that's the more common existing
    # case, fall back to str for a shelve-backed graph_pedia. Shared by
    # TokenizedDataset.__getitem__ and the graph-size/token-match filters in
    # run_seq2seq_train.py / run_seq2seq_eval.py, which index graph_pedia directly.
    try:
        return graph_pedia[graph_idx]
    except (KeyError, TypeError, AttributeError):
        # shelve (dbm.dumb, Python 3.7) doesn't raise KeyError/TypeError for a
        # non-str key -- shelve.Shelf.__getitem__ first tries self.cache[key] (a
        # plain dict; a KeyError there is caught internally by shelve itself, not
        # propagated), then falls through to self.dict[key.encode(...)], where an
        # int key raises AttributeError ('int' object has no attribute 'encode')
        # instead. Confirmed via a real traceback during a ScienceBenchmark
        # fine-tuning smoke test. Broadened here rather than assuming shelve's
        # exception type across versions.
        return graph_pedia[str(graph_idx)]


class TokenizedDataset(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self, data_training_args, training_args, tokenizer, seq2seq_dataset=None, graph_pedia=None, ):
        
        self.args = data_training_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset
        self.graph_pedia = graph_pedia

        self.conv_sep = " || "

    def _get_graph_entry(self, graph_idx):
        return get_graph_entry(self.graph_pedia, graph_idx)

    def map_alias(self, example):
        alias_map = {}
        example_list = example.split(' ')
        for i, ex in enumerate(example_list):
            if ex.lower() == 'as' and i > 0 and i + 1 < len(example_list):
                alias_map[example_list[i + 1]] = example_list[i - 1]
        return alias_map

    def replace_alias(self, example, mapping):
        # Word-boundary-safe substitution. The original `ex.replace(k, v)` did a raw
        # substring replace across the *entire* query string -- fine for Spider's gold
        # SQL, which almost never uses single-letter aliases (it uses "T1"/"T2", rarely
        # colliding with substrings of other words), but silently corrupting for
        # ScienceBenchmark's gold SQL, which commonly does (p, s, t, i, c, ...). E.g.
        # alias "p" -> "projects" turned "temporary" into "temprojectsorary" (the "p" in
        # "temporary" got replaced too). Confirmed empirically: ~48% (144/299) of
        # ScienceBenchmark dev examples' labels were corrupted this way before this fix;
        # Spider's near-total lack of single-letter aliases is why this was never seen
        # there. \b regex word boundaries fix it without changing the intent (still
        # replaces every whole-word occurrence of the alias, still strips the "as <alias>"
        # clause) -- this feeds seq2seq_dataset labels (the loss target), not the
        # exact_match/exec metric, which is computed against the unmodified `query` field.
        ex = example
        for k, v in mapping.items():
            ex = re.sub(r'\b' + re.escape(k) + r'\b', v, ex)
            ex = re.sub(r'\s+as\s+' + re.escape(v) + r'\b', '', ex, flags=re.IGNORECASE)

        return ex

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]
        question_in = " ".join(raw_item['raw_question_toks'])
        struct_in_norm = re.sub('  +', ' ', self._get_graph_entry(raw_item['graph_idx'])['new_struct_in'])
        seq_in = "{} ; {}".format(question_in, struct_in_norm)


        tokenized_question_and_schemas = self.tokenizer(
            seq_in,
            max_length=self.args.max_source_length,
        )
        # remove alias of sqls to reduce the training budget:
        alias_map = self.map_alias(raw_item['seq_out'])
        sql_norm_db_id = self.replace_alias(raw_item['seq_out'], alias_map)
        tokenized_inferred = self.tokenizer(
            sql_norm_db_id,
            max_length=self.args.max_target_length,
        )
        tokenized_question_and_schemas_input_ids = tokenized_question_and_schemas.data["input_ids"]
        tokenized_question_and_schemas_attention_mask = tokenized_question_and_schemas.data["attention_mask"]
        tokenized_inferred_input_ids = tokenized_inferred.data["input_ids"]
        
        item = {
            'input_ids': tokenized_question_and_schemas_input_ids,
            'attention_mask': tokenized_question_and_schemas_attention_mask,
            'labels': tokenized_inferred_input_ids,
        }
        # Add task name.
        if 'task_id' in raw_item:
            item['task_ids'] = raw_item['task_id']
        
        item['graph_idx'] = raw_item['graph_idx']
            
        
        # assertion:
        if len([a for a in tokenized_question_and_schemas.input_ids if a > 1]) != self._get_graph_entry(item['graph_idx'])['graph'].number_of_nodes():
            print('index: {}'.format(index))
            print('len of tokens is {}'.format(len([a for a in tokenized_question_and_schemas.input_ids if a > 1])))
            print('len of nodes is {}'.format(self._get_graph_entry(item['graph_idx'])['graph'].number_of_nodes()))

        return item

    def __len__(self):
        return len(self.seq2seq_dataset)