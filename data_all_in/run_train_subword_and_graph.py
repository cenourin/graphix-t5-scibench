import json
import pickle
import shelve
import argparse
import time
from transformers import AutoTokenizer

from map_subword_serialize import question_subword_matrix, schema_subword_matrix, schema_linking_subword
from Graph_Processing import SubwordGraphProcessor

'''
Replaces, for the training split only, the four separate scripts
(map_subword_question.py -> map_subword_schema.py -> map_subword_schema_linking.py -> Graph_Processing.py),
which each pickle.load()/pickle.dump() the *entire* split and therefore hold every example's
subword matrices in memory simultaneously across three full passes. Spider's training split
(8577 examples) does not fit that access pattern in 27GB RAM (confirmed OOM-killed, see kern.log).

This script performs the exact same per-example transformations, in the exact same order,
via the exact same functions -- but as a single streaming pass per example, dropping each
example's large intermediate fields immediately (mirroring what Graph_Processing.py already
does at the end of its own loop). That bounded per-example peak was measured at ~12GB on
Spider's train split -- fine there, but ScienceBenchmark's larger individual schemas (oncomx,
sdss) push the *finished* per-example graph objects themselves to be much bigger, so
accumulating all 4732 of them in a single in-memory dict before one final pickle.dump() (the
original design here) pushed host RSS to within a few MB of a 22GB container limit and then
to a near system-wide OOM on a second attempt, on a 27GB host, before any of the actual dump
happened -- confirmed via free -h during a real run of this script.

graph_pedia is now written incrementally to a shelve (on-disk dict-like store, keyed by
str(idx)) instead of accumulated in a Python dict: each `graph_pedia[key] = value` assignment
is pickled and flushed to disk immediately by the stdlib shelve/dbm layer (writeback=False,
the default), so this process's own memory footprint no longer grows with the number of
examples processed -- only with the size of whichever single example is currently in flight.
Consumers open this the same way (dict-like `[]` access), just via `shelve.open(path, "r")`
instead of `pickle.load()`; seq2seq/utils/dataset_graph.py::TokenizedDataset already falls
back to a str(graph_idx) key lookup if the int key isn't found, so it works against either a
plain pickled dict (existing dev/Spider graph_pedia files, unchanged) or this shelve store.
'''

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--syntax_path', type=str, required=True)
    arg_parser.add_argument('--dataset_path', type=str, required=True)
    arg_parser.add_argument('--table_path', type=str, required=True)
    arg_parser.add_argument('--plm', type=str, required=True)
    arg_parser.add_argument('--output_path', type=str, required=True, help='final seq2seq_train_dataset.json path')
    arg_parser.add_argument('--graph_output_path', type=str, required=True, help='graph_pedia_train.bin path')
    args = arg_parser.parse_args()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.plm, use_fast=True)
    tokenizer.add_tokens([' <', ' <='])

    tables = pickle.load(open(args.table_path, "rb"))
    seq2seq_dataset = json.load(open(args.dataset_path, "r"))
    syntax_dataset = json.load(open(args.syntax_path, "r"))

    processor = SubwordGraphProcessor()
    # flag="n": always start a fresh store (matches the old dict's "start empty" semantics
    # and avoids silently mixing in stale entries from a previous, possibly-interrupted run).
    graph_pedia = shelve.open(args.graph_output_path, flag="n")
    seq2seq_dataset_formal = []

    for i_str, data in seq2seq_dataset.items():
        idx = int(i_str)

        # --- map_subword_question.py ---
        processed_question_toks = data['raw_question_toks']
        relations = syntax_dataset[idx]['relations']
        question_sub_matrix, question_subword_dict = question_subword_matrix(
            processed_question_toks=processed_question_toks, relations=relations, tokenizer=tokenizer)
        data['question_subword_matrix'], data['question_subword_dict'] = question_sub_matrix, question_subword_dict
        data['question_token_relations'] = relations
        data['schema_linking'] = syntax_dataset[idx]['schema_linking']
        data['graph_idx'] = syntax_dataset[idx]['graph_idx']

        # --- map_subword_schema.py ---
        table_items = data['db_table_names']
        column_items = data['db_column_names']['column_name']
        db_id = data['db_id']
        db_sep = data['serialized_schema']
        schema_relations = tables[db_id]
        subword_matrix, subword_mapping_dict, new_struct_in, schema_to_ids = schema_subword_matrix(
            db_sep=db_sep, table_items=table_items, tokenizer=tokenizer,
            column_items=column_items, init_idx=0, tables=tables)
        data['schema_subword_relations'] = subword_matrix
        data['schema_relations'] = schema_relations
        data['schema_subword_mapping_dict'] = subword_mapping_dict
        data['new_struct_in'] = new_struct_in
        data['schema_to_ids'] = schema_to_ids

        # --- map_subword_schema_linking.py ---
        question_subword_len = len(data['question_subword_matrix'])
        schema_subword_len = len(data['schema_subword_relations'])
        schema_linking_subwords = schema_linking_subword(
            question_subword_dict=data['question_subword_dict'],
            schema_2_ids=data['schema_to_ids'],
            question_subword_len=question_subword_len,
            schema_subword_len=schema_subword_len,
            schema_linking=data['schema_linking'])
        data['schema_linking_subword'] = schema_linking_subwords

        # --- Graph_Processing.py ---
        new_data = processor.process_subgraph_utils(data)
        graph_pedia[str(idx)] = data['graph']  # shelve requires str keys

        del new_data['question_subword_matrix']
        del new_data['question_subword_dict']
        del new_data['question_token_relations']
        del new_data['schema_linking']
        del new_data['schema_subword_relations']
        del new_data['schema_relations']
        del new_data['schema_subword_mapping_dict']
        del new_data['schema_to_ids']
        del new_data['schema_linking_subword']
        del new_data['graph']
        seq2seq_dataset_formal.append(data)

        if idx % 1000 == 0:
            print("processing {}th data".format(idx))

    json.dump(seq2seq_dataset_formal, open(args.output_path, "w"))
    graph_pedia.close()

    print('Dataset preprocessing costs %.4fs .' % (time.time() - t0))


if __name__ == '__main__':
    main()
