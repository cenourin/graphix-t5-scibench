#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_file="$project_dir/.env"

if [[ ! -f "$env_file" ]]; then
  printf 'Configuracao ausente. Execute primeiro: %s/configurar_openwebui.sh\n' "$project_dir" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$env_file"
exec python3 "$project_dir/schemagraphsql_sqlite.py" "$@"
