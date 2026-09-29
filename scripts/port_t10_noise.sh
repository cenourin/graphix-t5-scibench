#!/usr/bin/env bash
# Noise envelope for T10 and the T7 resume runs (PORTABILIDADE.md, A7). GTX 1070.
# Same setup as scripts/port_t10_training.py (run its `prepare` first, and the T10 legacy and
# modern runs), then:
#   legacy2    the legacy run again           -> legacy run-to-run noise
#   modern2    the modern run again           -> modern run-to-run noise (also T7's A')
#   perturbed  modern, initial weights x (1 + 1e-6 * N(0,1)), seed 0
#              -> how much 1e-6 relative perturbations grow over 50 steps
#   T7 B       modern stopped after step 30 with a checkpoint, then resumed to step 50
set -u
cd "$(dirname "$0")/.."
T=data_all_in/data/port_tests/T10; T7=data_all_in/data/port_tests/T7; SP=data_all_in/data
mkdir -p $T/legacy2 $T/modern2 $T/perturbed $T7/A $T7/A2 $T7/B
M="-v $PWD/seq2seq:/app/seq2seq:ro -v $PWD/graphix_modern:/app/graphix_modern:ro -v $PWD/scripts:/app/scripts:ro -v $PWD/tests:/app/tests:ro -v $PWD/configs:/app/configs:ro -v $PWD/data_all_in:/app/data_all_in -w /app"
E="--user $(id -u):$(id -g) -e HOME=/tmp -e DGLBACKEND=pytorch -e HF_HOME=/tmp/hf -e HF_MODULES_CACHE=/tmp/hf/modules -e HF_DATASETS_CACHE=/tmp/hf/datasets -e GRAPHIX_TRAIN_DATASET_PATH=$SP/splits/spider_train.json -e GRAPHIX_TRAIN_GRAPH_PEDIA_PATH=$SP/output/graph_pedia_total.bin -e GRAPHIX_EVAL_DATASET_PATH=$SP/splits/spider_val.json -e GRAPHIX_EVAL_GRAPH_PEDIA_PATH=$SP/output/graph_pedia_total.bin -e GRAPHIX_MAX_GRAPH_NODES=512 -e HARNESS_NO_DROPOUT=1"
# expandable_segments: allocator setting only (the 1070 fragments at 8 GB; see PORTABILIDADE.md)
ME="$E -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -v $PWD/third_party:/app/third_party:ro"
MODERN=silveirabruno/graphix-modern@sha256:371f61af7521069e62bb8973c2c42584377659440b5902d025876fb94d859037  # a5
LEGACY=eyuansu62/graphix-text-to-sql:v2

FAIL=0
rc() { echo "$2 exit=$1"; [ "$1" = 0 ] || FAIL=1; }
cfg() {  # cfg SRC_ENV OUTPUT_DIR JSON_OVERRIDES DEST
  python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['output_dir']=sys.argv[2]; c.update(json.loads(sys.argv[3])); json.dump(c,open(sys.argv[4],'w'),indent=1)" \
    "$T/$1/config.json" "$2" "$3" "$4"
}

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp $M $MODERN python -c "
import torch; torch.manual_seed(0)
s = torch.load('$T/init_state_dict.bin', map_location='cpu', weights_only=True)
torch.save({k: v * (1 + 1e-6 * torch.randn_like(v)) for k, v in s.items()}, '$T/init_state_dict_perturbed.bin')"

cfg legacy $T/legacy2/run '{}' $T/legacy2/config.json
docker run --rm --gpus all $E -e GRAPHIX_INIT_STATE_DICT=$T/init_state_dict.bin -e HARNESS_ORDER_LOG=$T/legacy2/order.json $M $LEGACY \
  python scripts/port_entry_harness.py legacy $T/legacy2/config.json > $T/legacy2/train.log 2>&1; rc $? "legacy2"

cfg modern $T/modern2/run '{}' $T/modern2/config.json
docker run --rm --gpus all $ME -e GRAPHIX_INIT_STATE_DICT=$T/init_state_dict.bin -e HARNESS_ORDER_LOG=$T/modern2/order.json $M $MODERN \
  python scripts/port_entry_harness.py modern $T/modern2/config.json > $T/modern2/train.log 2>&1; rc $? "modern2"

cfg modern $T/perturbed/run '{}' $T/perturbed/config.json
docker run --rm --gpus all $ME -e GRAPHIX_INIT_STATE_DICT=$T/init_state_dict_perturbed.bin -e HARNESS_ORDER_LOG=$T/perturbed/order.json $M $MODERN \
  python scripts/port_entry_harness.py modern $T/perturbed/config.json > $T/perturbed/train.log 2>&1; rc $? "perturbed"

rm -rf $T7/A/run $T7/A2/run
cp -r $T/modern/run $T7/A/run
cp -r $T/modern2/run $T7/A2/run
cfg modern $T7/B/run '{}' $T7/B/config.json
docker run --rm --gpus all $ME -e GRAPHIX_INIT_STATE_DICT=$T/init_state_dict.bin -e HARNESS_STOP_AT_STEP=30 $M $MODERN \
  python scripts/port_entry_harness.py modern $T7/B/config.json > $T7/B/train_part1.log 2>&1; rc $? "B part1"
python3 -c "import json; c=json.load(open('$T7/B/config.json')); c['overwrite_output_dir']=False; json.dump(c,open('$T7/B/config_resume.json','w'),indent=1)"
docker run --rm --gpus all $ME -e GRAPHIX_INIT_STATE_DICT=$T/init_state_dict.bin $M $MODERN \
  python scripts/port_entry_harness.py modern $T7/B/config_resume.json > $T7/B/train_part2.log 2>&1; rc $? "B part2"
exit $FAIL
