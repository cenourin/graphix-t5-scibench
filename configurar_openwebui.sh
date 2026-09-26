#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_file="$project_dir/.env"

read -r -p "URL do Open WebUI [http://home2.scanuto.com:8081]: " openwebui_url
openwebui_url="${openwebui_url:-http://home2.scanuto.com:8081}"
read -r -p "ID do modelo [qwen3-coder-next:latest]: " openwebui_model
openwebui_model="${openwebui_model:-qwen3-coder-next:latest}"
read -r -s -p "API key do Open WebUI (entrada oculta): " openwebui_key
printf '\n'

if [[ -z "$openwebui_key" ]]; then
  printf 'Erro: a API key nao pode ficar vazia.\n' >&2
  exit 1
fi
if [[ "$openwebui_url" == *$'\n'* || "$openwebui_model" == *$'\n'* || "$openwebui_key" == *$'\n'* ]]; then
  printf 'Erro: os valores nao podem conter quebra de linha.\n' >&2
  exit 1
fi

umask 077
{
  printf 'export OPENWEBUI_URL=%q\n' "$openwebui_url"
  printf 'export OPENWEBUI_API_KEY=%q\n' "$openwebui_key"
  printf 'export OPENWEBUI_MODEL=%q\n' "$openwebui_model"
} > "$env_file"
chmod 600 "$env_file"
unset openwebui_key

printf 'Configuracao salva em %s (permissao 600).\n' "$env_file"
printf 'Use ./executar_schemagraphsql.sh para executar o programa.\n'
