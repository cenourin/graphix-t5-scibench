db_dir='data_all_in/data/sciencebenchmark/database'

train_mode='train'
train_data='data_all_in/data/sciencebenchmark/output/train_merged.json'
tables_data='data_all_in/data/sciencebenchmark/output/tables_merged.json'
tables_out='data_all_in/data/sciencebenchmark/output/tables.bin'
train_out='data_all_in/data/sciencebenchmark/output/train.bin'
syntax_train_out='data_all_in/data/sciencebenchmark/output/train_syntax.json'

eval_mode='dev'
eval_data='data_all_in/data/sciencebenchmark/output/dev_merged.json'
eval_out='data_all_in/data/sciencebenchmark/output/dev.bin'
syntax_eval_out='data_all_in/data/sciencebenchmark/output/dev_syntax.json'

echo "merge ScienceBenchmark train/dev + tables (cordis+oncomx+sdss)"
python3 -u data_all_in/merge_sciencebenchmark.py \
  --train_output_path ${train_data} --dev_output_path ${eval_data}
python3 -c "
import json
merged = []
for db in ['cordis','oncomx','sdss']:
    merged.extend(json.load(open(f'data_all_in/data/sciencebenchmark/{db}/tables.json')))
json.dump(merged, open('${tables_data}','w'), indent=4)
"

# contextual semantic match -- NOTE: no --skip_large here (oncomx has 112
# columns, over the 100-column safety threshold; see project decision to
# keep all 3 DBs rather than silently drop oncomx).
echo "Starting to preprocess the basic ScienceBenchmark train dataset"
python3 -u data_all_in/preprocess/process_dataset.py --db_dir ${db_dir} --dataset_path ${train_data} --raw_table_path ${tables_data} --table_path ${tables_out} \
--output_path ${train_out}

echo "Starting to preprocess the ScienceBenchmark train dataset for training..."
python3 -u data_all_in/preprocess/inject_syntax.py --dataset_path ${train_out} --mode ${train_mode} --output_path ${syntax_train_out}

echo "Starting to preprocess the basic ScienceBenchmark dev dataset"
python3 -u data_all_in/preprocess/process_dataset.py --db_dir ${db_dir} --dataset_path ${eval_data} --table_path ${tables_out} \
--output_path ${eval_out}

echo "Starting to preprocess the ScienceBenchmark eval dataset for dev..."
python3 -u data_all_in/preprocess/inject_syntax.py --dataset_path ${eval_out} --mode ${eval_mode} --output_path ${syntax_eval_out} --dev_graph_idx_offset 0
