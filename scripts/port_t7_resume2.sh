#!/usr/bin/env bash
# T7 runs that isolate the resume (PORTABILIDADE.md, A6.3). Needs T7/B from
# scripts/port_t10_noise.sh (B stopped at step 30 with checkpoint-30, then resumed to 50).
#   D  a second resume from the same checkpoint-30, logging the training-item order
#   E  a resume from checkpoint-30 with the Adafactor state emptied (negative control)
set -u
cd "$(dirname "$0")/.."
T=data_all_in/data/port_tests/T10; T7=data_all_in/data/port_tests/T7; SP=data_all_in/data
M="-v $PWD/seq2seq:/app/seq2seq:ro -v $PWD/graphix_modern:/app/graphix_modern:ro -v $PWD/scripts:/app/scripts:ro -v $PWD/tests:/app/tests:ro -v $PWD/configs:/app/configs:ro -v $PWD/third_party:/app/third_party:ro -v $PWD/data_all_in:/app/data_all_in -w /app"
E="--user $(id -u):$(id -g) -e HOME=/tmp -e DGLBACKEND=pytorch -e HF_HOME=/tmp/hf -e HF_MODULES_CACHE=/tmp/hf/modules -e HF_DATASETS_CACHE=/tmp/hf/datasets -e GRAPHIX_TRAIN_DATASET_PATH=$SP/splits/spider_train.json -e GRAPHIX_TRAIN_GRAPH_PEDIA_PATH=$SP/output/graph_pedia_total.bin -e GRAPHIX_EVAL_DATASET_PATH=$SP/splits/spider_val.json -e GRAPHIX_EVAL_GRAPH_PEDIA_PATH=$SP/output/graph_pedia_total.bin -e GRAPHIX_MAX_GRAPH_NODES=512 -e HARNESS_NO_DROPOUT=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e GRAPHIX_INIT_STATE_DICT=$T/init_state_dict.bin"
FAIL=0
rc() { echo "$2 exit=$1"; [ "$1" = 0 ] || FAIL=1; }
for run in D E; do
  rm -rf $T7/$run && mkdir -p $T7/$run/run
  cp -r $T7/B/run/checkpoint-30 $T7/$run/run/
  # E: keep optimizer.pt (the port, like 4.17, loads optimizer and scheduler only together)
  # but empty its per-parameter state, so only the Adafactor moments are lost, not the LR.
  [ $run = E ] && docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp $M silveirabruno/graphix-modern@sha256:371f61af7521069e62bb8973c2c42584377659440b5902d025876fb94d859037 python -c "
import torch; p='$T7/$run/run/checkpoint-30/optimizer.pt'
sd = torch.load(p, map_location='cpu', weights_only=True); n = len(sd['state']); sd['state'] = {}; torch.save(sd, p)
print('E: emptied Adafactor state of', n, 'parameters')"
  python3 -c "import json,sys; c=json.load(open('$T7/B/config_resume.json')); c['output_dir']='$T7/$run/run'; json.dump(c,open('$T7/$run/config.json','w'),indent=1)"
  docker run --rm --gpus all $E -e HARNESS_ORDER_LOG=$T7/$run/order.json $M silveirabruno/graphix-modern@sha256:371f61af7521069e62bb8973c2c42584377659440b5902d025876fb94d859037 \
    python scripts/port_entry_harness.py modern $T7/$run/config.json > $T7/$run/train.log 2>&1
  rc $? "$run"
done
exit $FAIL
