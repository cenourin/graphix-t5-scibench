train_data='data_all_in/data/spider/train.json'
tables_data='data_all_in/data/spider/tables.json'
tables_out='data_all_in/data/tables.bin'
train_out='data_all_in/data/train.bin'
syntax_train_out='data_all_in/data/train_syntax.json'
train_sampling_out='data_all_in/data/train_sampling.json'
configs='configs/data_pre_train.json'
seq2seq_train_dataset='data_all_in/data/output/seq2seq_train_dataset_pre.json'
seq2seq_train_dataset_final='data_all_in/data/output/seq2seq_train_dataset.json'
graph_output_train_path='data_all_in/data/output/graph_pedia_train.bin'


# serialize databases
echo "Starting to serialize databases ..."
python3 seq2seq/run_peteshaw_train.py ${configs}

# question + schema + schema-linking subword injection, then graph construction,
# all in a single streaming pass (see data_all_in/run_train_subword_and_graph.py
# for why: the original 4-separate-scripts pipeline OOMs on the full training
# split because each script pickles the entire split into memory at once).
echo "Starting to split relations into subwords and generate graph examples ..."
python3 -u data_all_in/run_train_subword_and_graph.py --syntax_path ${syntax_train_out} --dataset_path ${seq2seq_train_dataset} \
--table_path ${tables_out} --plm data_all_in/t5-large --output_path ${seq2seq_train_dataset_final} \
--graph_output_path ${graph_output_train_path}

