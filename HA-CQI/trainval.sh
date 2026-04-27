#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# HA-CQI 当前主线只保留 SAR 洪水变化检测训练入口。
# 详细参数可通过环境变量覆盖，默认值见 trainval_s1gfloods.sh。
bash trainval_s1gfloods.sh "$@"
