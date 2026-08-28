#!/bin/bash
# Downloads the ScienceBenchmark dataset (https://github.com/ckosten/sciencebenchmark_dataset)
# into data_all_in/data/sciencebenchmark/{cordis,oncomx,sdss}/, mirroring the layout used
# for Spider under data_all_in/data/spider/.
#
# ScienceBenchmark covers 3 real-world scientific databases (not 166 like Spider):
#   - cordis  : EU research project metadata   (db_id: cordis_temporary)
#   - oncomx  : biomedical / cancer knowledge base (db_id: oncomx_v1_0_25_small)
#   - sdss    : astronomy (Sloan Digital Sky Survey) (db_id: skyserver_dr16_2020_11_30)
#
# Each folder gets: dev.json, seed.json (human-written), synth.json (template-generated),
# tables.json (schema, same field names as Spider's tables.json: table_names_original,
# column_names_original, column_types, foreign_keys, primary_keys). The "sql" field inside
# the examples already matches Spider's process_sql.py parse-tree format.
#
# IMPORTANT — known gap, not solved by this script:
# The actual databases here ship ONLY as PostgreSQL dumps (see POSTGRES_DUMPS below), but
# data_all_in/preprocess/{common_utils.py,common_utils_proton.py} hardcode sqlite3 and
# expect <db_dir>/<db_id>/<db_id>.sqlite (see process_dataset.py --db_dir). Before this data
# can flow through the existing run_pre.sh pipeline, the 3 Postgres dumps restored below
# must be converted to SQLite files at:
#   data_all_in/data/sciencebenchmark/database/<db_id>/<db_id>.sqlite
# (e.g. via pgloader, or pg_dump --inserts + a Postgres->SQLite type mapping). That
# conversion is intentionally NOT done here since it needs a running Postgres server.

set -euo pipefail

REPO_RAW="https://raw.githubusercontent.com/ckosten/sciencebenchmark_dataset/master"
OUT_DIR="data_all_in/data/sciencebenchmark"
DUMP_DIR="${OUT_DIR}/postgres_dumps"

declare -A DB_IDS=(
  [cordis]="cordis_temporary"
  [oncomx]="oncomx_v1_0_25_small"
  [sdss]="skyserver_dr16_2020_11_30"
)

# Google Drive file IDs for the PostgreSQL dumps, from the repo's README.
declare -A DRIVE_IDS=(
  [cordis]="1-YXF9mPN7GmK4ApBz2aZjAoJkdpPE6JF"
  [sdss]="1-11YVe1dsqh-x3RoVPQP8aKQouBESRm9"
  [oncomx]="1Y02tXXzaWaYwSKQtnWbYPe7dKk0T5Xmp"
)

echo "== Downloading ScienceBenchmark JSON splits + schemas =="
for db in cordis oncomx sdss; do
  mkdir -p "${OUT_DIR}/${db}"
  for f in dev.json seed.json synth.json tables.json; do
    dest="${OUT_DIR}/${db}/${f}"
    if [ -s "${dest}" ]; then
      echo "  skip ${dest} (already exists)"
      continue
    fi
    echo "  fetching ${db}/${f}"
    curl -sSL -o "${dest}" "${REPO_RAW}/${db}/${f}"
  done
done

echo "== Downloading PostgreSQL dumps (via gdown, large files) =="
# The host Python is externally-managed (PEP 668), so gdown lives in a local venv
# (.venv-tools/), same convention already used elsewhere in this checkout.
VENV_DIR=".venv-tools"
if [ ! -x "${VENV_DIR}/bin/gdown" ]; then
  echo "  setting up ${VENV_DIR} with gdown"
  python3 -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install -q --upgrade pip gdown
fi

mkdir -p "${DUMP_DIR}"
for db in cordis oncomx sdss; do
  dest="${DUMP_DIR}/${db}.dump"
  if [ -s "${dest}" ]; then
    echo "  skip ${dest} (already exists)"
    continue
  fi
  echo "  fetching ${db} postgres dump (id=${DRIVE_IDS[$db]})"
  "${VENV_DIR}/bin/gdown" "${DRIVE_IDS[$db]}" -O "${dest}"
done

cat <<EOF

Done. Layout:
  ${OUT_DIR}/{cordis,oncomx,sdss}/{dev,seed,synth,tables}.json
  ${DUMP_DIR}/{cordis,oncomx,sdss}.dump   (pg_dump tar-format archives, pg_restore-able directly)

Next steps (manual, not automated by this script):
  1. Restore each dump into a PostgreSQL 9.5+ server. Before restoring, run
     'CREATE EXTENSION pg_trgm;' in psql (required by these dumps).
       pg_restore -d <target_db> ${DUMP_DIR}/<db>.dump
  2. Convert each restored database to SQLite at
       ${OUT_DIR}/database/<db_id>/<db_id>.sqlite
     (db_ids: ${DB_IDS[cordis]}, ${DB_IDS[oncomx]}, ${DB_IDS[sdss]})
     since data_all_in/preprocess/common_utils*.py only knows how to open sqlite3 files.
  3. Only after step 2 can seed.json/synth.json/dev.json be run through
     data_all_in/preprocess/process_dataset.py (--db_dir ${OUT_DIR}/database),
     analogous to data_all_in/run/run_syntax.sh for Spider.
EOF
