#!/usr/bin/env bash
set -euo pipefail
: "${TYPESAFE_API_KEY:?请先设置 TYPESAFE_API_KEY}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
curl --fail-with-body --silent --show-error --max-time 15 \
  https://api.typesafe.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary "@$script_dir/request.json" \
  -o "$script_dir/response.json"
printf '响应已保存到 %s/response.json\n' "$script_dir"
