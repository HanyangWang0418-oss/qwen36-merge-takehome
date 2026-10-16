#!/usr/bin/env bash
# Mirror the 51 SWE-bench Pro v2 HARD task images from ghcr.io into Alibaba Cloud ACR.
# Run on the ECS build box (docker + skopeo installed), not on a laptop.
#
#   export ACR_REGISTRY=<instance>.cn-hangzhou.personal.cr.aliyuncs.com/<namespace>
#   docker login $ACR_REGISTRY            # or: skopeo login
#   scripts/mirror_images.sh [--parallel 4]
#
# Idempotent: skips images whose digest already matches. Writes per-image source and
# destination digests to data/manifests/images_acr_swebench_pro.json via scripts/lock_images.py.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ACR_REGISTRY:?set ACR_REGISTRY, e.g. xxx.cn-hangzhou.personal.cr.aliyuncs.com/qwen36}"
PAR=4; [ "${1:-}" = "--parallel" ] && PAR=$2
SRC_REPO=ghcr.io/scaleapi/swe-bench_pro-v2
DST_REPO=$ACR_REGISTRY/swe-bench_pro-v2
IDS=$(python3 -c "import json;print('\n'.join(json.load(open('data/manifests/swebench_pro_v2_hard.json'))['task_ids']))")
mkdir -p logs/images
mirror_one() {
  id=$1
  src="docker://$SRC_REPO:$id"; dst="docker://$DST_REPO:$id"
  sd=$(skopeo inspect --raw "$src" 2>/dev/null | sha256sum | cut -d' ' -f1 || true)
  dd=$(skopeo inspect --raw "$dst" 2>/dev/null | sha256sum | cut -d' ' -f1 || true)
  if [ -n "$dd" ] && [ "$sd" = "$dd" ]; then echo "skip  $id (already mirrored)"; return 0; fi
  echo "copy  $id"
  # --all keeps every platform in the manifest list; retries cover ghcr rate limits
  skopeo copy --all --retry-times 5 "$src" "$dst" > "logs/images/mirror_$id.log" 2>&1 && echo "done  $id" || { echo "FAIL  $id (see logs/images/mirror_$id.log)"; return 1; }
}
export -f mirror_one; export SRC_REPO DST_REPO
echo "$IDS" | xargs -P "$PAR" -I{} bash -c 'mirror_one {}'
uv run --no-sync python scripts/lock_images.py swebench_pro
