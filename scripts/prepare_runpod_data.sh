#!/usr/bin/env bash
# Unversioned data MODERN_A needs on the RunPod network volume, sent straight from where it
# lives in this checkout: no local tar and no staging copy (only a small sha256 manifest is
# written locally, into data_all_in/data/manifests/, which is gitignored).
#
#   scripts/prepare_runpod_data.sh list                    # items, sizes, pod paths, total (default)
#   scripts/prepare_runpod_data.sh manifest                # sha256 of every file -> data_all_in/data/manifests/
#   scripts/prepare_runpod_data.sh s3 VOLUME_ID [REGION]   # RunPod S3-compatible API (no pod needed)
#   scripts/prepare_runpod_data.sh rsync  HOST PORT        # resumable upload over SSH (rsync on both ends)
#   scripts/prepare_runpod_data.sh stream HOST PORT        # fallback: tar | zstd | ssh (not resumable)
#   scripts/prepare_runpod_data.sh verify HOST PORT        # sha256sum -c on the pod
#
# Layout. The code reads data relative to the repository root, as data_all_in/data/... and
# data_all_in/t5-base-st. On the pod, /workspace/data holds the CONTENTS of data_all_in/data
# (same subpaths: spider/, sciencebenchmark/, graph_export/, splits/, output/, ...) plus
# t5-base-st/ and manifests/. The repository clone then needs two symlinks, no code change:
#   ln -sfn /workspace/data            /workspace/project/graphix-t5-scibench/data_all_in/data
#   ln -sfn /workspace/data/t5-base-st /workspace/project/graphix-t5-scibench/data_all_in/t5-base-st
# (data_all_in/data is gitignored; t5-base-st shows as untracked in git status.)
# RunPod S3 API (used for the 2026-09-29 transfer, volume in EU-RO-1, endpoint
# https://s3api-eu-ro-1.runpod.io, 23.02 GB / 450 files): the endpoint rejects parts of 256 MB
# (HTTP 413) and, with the CLI default of 8 MB parts (~1800 parts for the 15.1 GB skyserver
# file), lost parts before CompleteMultipartUpload ("InvalidPart ... parts missing"). What
# worked: 64 MB parts and 4 parallel requests, set in the AWS profile (not here):
#   aws configure set s3.multipart_chunksize 64MB --profile runpod
#   aws configure set s3.max_concurrent_requests 4 --profile runpod
# plus --cli-read-timeout 3600 for the single large file, so a slow CompleteMultipartUpload is
# not retried. After a failed multipart upload, abort it (s3api list-multipart-uploads /
# abort-multipart-upload) before retrying. Credentials stay in ~/.aws (never in this repo).
# INCLUDE_T10=1 adds the T10 inputs (initial weights + the 1070 reference curves), needed
# only to replay T10 exactly on the 4090 (50 steps comparable step by step with the 1070).
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."
REPO=$PWD
REMOTE=${REMOTE_DATA_DIR:-/workspace/data}
INCLUDE_T10=${INCLUDE_T10:-0}
MANIFEST_DIR=data_all_in/data/manifests
MANIFEST=$MANIFEST_DIR/DATA_MANIFEST.sha256

# pod path (relative to /workspace/data) | benchmark | why. Local source: data_all_in/data/<path>,
# except t5-base-st (local: data_all_in/t5-base-st).
ITEMS=(
  "splits/spider_train.json|spider|study train split (train-carved)"
  "splits/spider_val.json|spider|study validation split"
  "output/seq2seq_dev_dataset.json|spider|official dev (final dev eval)"
  "train_syntax.json|spider|HF loader, train split (seq2seq/datasets/spider)"
  "dev_syntax.json|spider|HF loader, dev split"
  "spider/database|spider|166 SQLite DBs: schema+content serialization, execution eval"
  "graph_export/spider_total|spider|A0 graph export (replaces graph_pedia_total.bin)"
  "splits/sciencebenchmark_train.json|sciencebenchmark|study train split"
  "splits/sciencebenchmark_val.json|sciencebenchmark|study validation split"
  "sciencebenchmark/output/seq2seq_dev_dataset.json|sciencebenchmark|official dev (final dev eval)"
  "sciencebenchmark/output/train_syntax.json|sciencebenchmark|HF loader, train split"
  "sciencebenchmark/output/dev_syntax.json|sciencebenchmark|HF loader, dev split"
  "sciencebenchmark/output/tables_merged.json|sciencebenchmark|HF loader schema file"
  "sciencebenchmark/database|sciencebenchmark|3 SQLite DBs (cordis, oncomx, skyserver 15 GB)"
  "graph_export/sciencebenchmark_train|sciencebenchmark|A0 graph export, train (also val)"
  "graph_export/sciencebenchmark_dev|sciencebenchmark|A0 graph export, dev"
  "graph_export/manifest.json|both|A0 manifest: maps GRAPHIX_*_GRAPH_PEDIA_PATH to the exports"
  "graph_export/T2_report.json|both|A0/T2 validation record (provenance)"
  "t5-base-st|both|t5-base as safetensors (MODERN_A cannot torch.load the .bin)"
)
if [ "$INCLUDE_T10" = 1 ]; then
  ITEMS+=(
    "port_tests/T10/init_state_dict.bin|spider|T10 initial weights (exact 50-step replay vs the 1070)"
    "port_tests/T10/T10_report.json|spider|T10 envelope (reference)"
    "port_tests/T10/modern/config.json|spider|T10 modern config (reference)"
    "port_tests/T10/modern/order.json|spider|T10 item order (reference)"
  )
  for r in legacy legacy2 legacy3 modern modern2 modern3; do
    ITEMS+=("port_tests/T10/$r/run/trainer_state.json|spider|T10 loss curve $r (reference)")
  done
fi

src() { [ "$1" = t5-base-st ] && echo data_all_in/t5-base-st || echo "data_all_in/data/$1"; }
targets() { for it in "${ITEMS[@]}"; do echo "${it%%|*}"; done; }

list() {
  local total=0
  printf "%-10s %-17s %s\n" "MB" "benchmark" "current path -> pod path"
  for it in "${ITEMS[@]}"; do
    IFS='|' read -r t b why <<< "$it"
    local p; p=$(src "$t")
    [ -e "$p" ] || { echo "MISSING: $REPO/$p" >&2; exit 1; }
    local s; s=$(du -sbL "$p" | cut -f1); total=$((total + s))
    printf "%-10.1f %-17s %s -> %s/%s   (%s)\n" "$(echo "$s/1000000" | bc -l)" "$b" "$REPO/$p" "$REMOTE" "$t" "$why"
  done
  printf "TOTAL %.2f GB (%.2f GiB), %d items (+ %s)\n" "$(echo "$total/10^9" | bc -l)" "$(echo "$total/2^30" | bc -l)" \
    "${#ITEMS[@]}" "$MANIFEST"
}

# sha256 of every file, paths relative to /workspace/data (so `cd /workspace/data && sha256sum -c` works)
manifest() {
  mkdir -p "$MANIFEST_DIR"
  local tmp; tmp=$(mktemp)
  for t in $(targets); do
    if [ "$t" = t5-base-st ]; then (cd data_all_in && find -L t5-base-st -type f | sort | xargs -d '\n' sha256sum)
    else (cd data_all_in/data && find -L "$t" -type f | sort | xargs -d '\n' sha256sum); fi
  done > "$tmp"
  mv "$tmp" "$MANIFEST"
  echo "$(wc -l < "$MANIFEST") files -> $MANIFEST"
}

ssh_cmd() { echo "ssh -p $2 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"; }

do_s3() {
  # Credentials come from an AWS profile (default "runpod", created with `aws configure --profile
  # runpod`, stored in ~/.aws/credentials), never from the command line or the repository.
  # File-level resume: directories go through `aws s3 sync --size-only` and single files are
  # skipped when the object already exists with the same size, so a re-run only sends what is
  # missing or incomplete (an interrupted file is sent again whole). Content is checked
  # afterwards on the pod with sha256sum -c against the manifest.
  local vol=$1 region=${2:-eu-ro-1}
  local AWS=${AWS_BIN:-aws}
  command -v "$AWS" >/dev/null || { echo "aws CLI not found (AWS_BIN=$AWS)" >&2; exit 1; }
  export AWS_PROFILE=${AWS_PROFILE:-runpod}
  local EP=(--region "$region" --endpoint-url "https://s3api-$region.runpod.io")
  local prefix=${REMOTE#/workspace/}   # the volume root is the bucket: /workspace/data -> data/
  for t in $(targets) manifests; do
    local p; p=$(src "$t")
    if [ -d "$p" ]; then
      echo "== sync $p -> s3://$vol/$prefix/$t/"
      "$AWS" s3 sync "${EP[@]}" --size-only --no-progress "$p" "s3://$vol/$prefix/$t"
    else
      local local_size remote_size
      local_size=$(stat -Lc %s "$p")
      remote_size=$("$AWS" s3api head-object "${EP[@]}" --bucket "$vol" --key "$prefix/$t" \
                      --query ContentLength --output text 2>/dev/null || echo none)
      if [ "$remote_size" = "$local_size" ]; then echo "== skip $t (already uploaded, $local_size bytes)"; continue; fi
      echo "== cp $p -> s3://$vol/$prefix/$t ($local_size bytes)"
      "$AWS" s3 cp "${EP[@]}" --no-progress "$p" "s3://$vol/$prefix/$t"
    fi
  done
  echo "upload pass finished: s3://$vol/$prefix/ ; validate on the pod: cd $REMOTE && sha256sum --quiet -c manifests/DATA_MANIFEST.sha256"
}

do_rsync() {
  local host=$1 port=$2 SSH; SSH=$(ssh_cmd "$host" "$port")
  $SSH "root@$host" "command -v rsync >/dev/null" || { echo "no rsync on the pod: use 's3' or 'stream'" >&2; exit 1; }
  $SSH "root@$host" "mkdir -p $REMOTE"
  local zc="-z"
  rsync --version | grep -q zstd && $SSH "root@$host" "rsync --version | grep -q zstd" && zc="--compress --compress-choice=zstd"
  # "dir/./path" + -R recreates only "path" on the receiver; --partial --append-verify: resumable
  { for t in $(targets) manifests; do
      [ "$t" = t5-base-st ] && echo "data_all_in/./t5-base-st" || echo "data_all_in/data/./$t"; done; } |
    rsync -aRL --partial --append-verify --info=progress2 $zc --files-from=- -r -e "$SSH" ./ "root@$host:$REMOTE/"
  echo "uploaded; now: $0 verify $host $port"
}

do_stream() {
  local host=$1 port=$2 SSH; SSH=$(ssh_cmd "$host" "$port")
  $SSH "root@$host" "command -v zstd >/dev/null" || { echo "no zstd on the pod" >&2; exit 1; }
  local rels=(); for t in $(targets) manifests; do [ "$t" = t5-base-st ] || rels+=("$t"); done
  tar -chf - -C data_all_in/data "${rels[@]}" -C "$REPO/data_all_in" t5-base-st | zstd -T0 -3 |
    $SSH "root@$host" "mkdir -p $REMOTE && zstd -d | tar -xf - -C $REMOTE"
  echo "streamed; now: $0 verify $host $port"
}

do_verify() {
  local host=$1 port=$2 SSH; SSH=$(ssh_cmd "$host" "$port")
  $SSH "root@$host" "cd $REMOTE && sha256sum --quiet -c manifests/DATA_MANIFEST.sha256 && date -u > manifests/DATA_VERIFIED && echo DATA VERIFIED"
}

case "${1:-list}" in
  list) list ;;
  manifest) manifest ;;
  s3) [ -s "$MANIFEST" ] || manifest; do_s3 "$2" "${3:-eu-ro-1}" ;;
  rsync) [ -s "$MANIFEST" ] || manifest; do_rsync "$2" "$3" ;;
  stream) [ -s "$MANIFEST" ] || manifest; do_stream "$2" "$3" ;;
  verify) do_verify "$2" "$3" ;;
  *) echo "usage: $0 list|manifest|s3 VOLUME_ID [REGION]|rsync HOST PORT|stream HOST PORT|verify HOST PORT" >&2; exit 2 ;;
esac
