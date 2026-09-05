eval_data='data_all_in/data/sciencebenchmark/output/dev_merged.json'
tables_data='data_all_in/data/sciencebenchmark/output/tables_merged.json'
tables_out='data_all_in/data/sciencebenchmark/output/tables.bin'
eval_out='data_all_in/data/sciencebenchmark/output/dev.bin'
syntax_eval_out='data_all_in/data/sciencebenchmark/output/dev_syntax.json'
configs='configs/data_pre_dev_sciencebenchmark.json'
seq2seq_eval_dataset='data_all_in/data/sciencebenchmark/output/seq2seq_dev_dataset_pre.json'
seq2seq_eval_out1='data_all_in/data/sciencebenchmark/output/seq2seq_dev_dataset.bin'
seq2seq_eval_dataset_final='data_all_in/data/sciencebenchmark/output/seq2seq_dev_dataset.json'
graph_output_dev_path='data_all_in/data/sciencebenchmark/output/graph_pedia_dev.bin'


# serialize databases
# NOTE: run_peteshaw_dev.py hardcodes its output path to
# data_all_in/data/output/seq2seq_dev_dataset_pre.json regardless of config
# (shared with the Spider pipeline) -- copy it into our own output dir right after.
echo "Starting to serialize databases ..."
python3 seq2seq/run_peteshaw_dev.py ${configs}
cp data_all_in/data/output/seq2seq_dev_dataset_pre.json ${seq2seq_eval_dataset}

# question relation injection
echo "Starting to split question relations into subwords ..."
python3 -u data_all_in/map_subword_question.py --syntax_path ${syntax_eval_out} --dataset_path ${seq2seq_eval_dataset} \
--dataset_output_path ${seq2seq_eval_out1} --plm data_all_in/t5-large

# database relation injection
echo "Starting to split schema relations into subwords ..."
python3 -u data_all_in/map_subword_schema.py --dataset_path ${seq2seq_eval_out1} --dataset_output_path ${seq2seq_eval_out1} \
--plm data_all_in/t5-large --table_path ${tables_out}

# schema_linking relation injection
echo "Starting to split schema-linking relations into subwords ..."
python3 -u data_all_in/map_subword_schema_linking.py --dataset_path ${seq2seq_eval_out1} --dataset_output_path ${seq2seq_eval_out1}

# construct graph now:
echo "Starting to generate graph examples ..."
python3 -u data_all_in/Graph_Processing.py --dataset_path ${seq2seq_eval_out1} --output_path ${seq2seq_eval_dataset_final} \
--graph_output_path ${graph_output_dev_path}
