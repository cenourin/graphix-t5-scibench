train_data='data_all_in/data/sciencebenchmark/output/train_merged.json'
tables_data='data_all_in/data/sciencebenchmark/output/tables_merged.json'
tables_out='data_all_in/data/sciencebenchmark/output/tables.bin'
train_out='data_all_in/data/sciencebenchmark/output/train.bin'
syntax_train_out='data_all_in/data/sciencebenchmark/output/train_syntax.json'
configs='configs/data_pre_train_sciencebenchmark.json'
seq2seq_train_dataset='data_all_in/data/sciencebenchmark/output/seq2seq_train_dataset_pre.json'
seq2seq_train_dataset_final='data_all_in/data/sciencebenchmark/output/seq2seq_train_dataset.json'
graph_output_train_path='data_all_in/data/sciencebenchmark/output/graph_pedia_train.bin'


# serialize databases
# NOTE: run_peteshaw_train.py hardcodes its output path to
# data_all_in/data/output/seq2seq_train_dataset_pre.json regardless of config
# (shared with the Spider pipeline) -- copy it into our own output dir right after.
echo "Starting to serialize databases ..."
python3 seq2seq/run_peteshaw_train.py ${configs}
cp data_all_in/data/output/seq2seq_train_dataset_pre.json ${seq2seq_train_dataset}

# question + schema + schema-linking subword injection, then graph construction,
# all in a single streaming pass (same OOM-avoidance rationale as Spider's
# train split -- see data_all_in/run_train_subword_and_graph.py).
echo "Starting to split relations into subwords and generate graph examples ..."
python3 -u data_all_in/run_train_subword_and_graph.py --syntax_path ${syntax_train_out} --dataset_path ${seq2seq_train_dataset} \
--table_path ${tables_out} --plm data_all_in/t5-large --output_path ${seq2seq_train_dataset_final} \
--graph_output_path ${graph_output_train_path}
