import json
import pickle
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
does at the end of its own loop). Peak RSS measured on this access pattern for the full split
is ~12GB, comfortably within budget.
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
    graph_pedia = {}
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
        graph_pedia[idx] = data['graph']

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
    pickle.dump(graph_pedia, open(args.graph_output_path, "wb"))

    print('Dataset preprocessing costs %.4fs .' % (time.time() - t0))


if __name__ == '__main__':
    main()
