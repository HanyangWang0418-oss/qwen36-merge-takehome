#!/usr/bin/env bash
# Build the Terminal-Bench 2.0 task images at the locked dataset commit and push them to ACR
# under the exact names harbor's `ack` backend derives: {registry}/{task}:latest
# (src/harbor/environments/ack.py::_get_image_url). A second tag :<commit8> is pushed for provenance.
#
#   export ACR_REGISTRY=<instance>.cn-hangzhou.personal.cr.aliyuncs.com/<namespace>
#   docker login $ACR_REGISTRY
#   scripts/build_tb2_images.sh [--parallel 2] [--only task1,task2]
#
# Needs docker buildx on an amd64 build box (ECS). Compose-based tasks are listed and skipped
# (harbor ack runs single-container pods); they are handled separately, see docs/infra_aliyun.md.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ACR_REGISTRY:?set ACR_REGISTRY}"
PAR=2; ONLY=""
while [ $# -gt 0 ]; do case $1 in --parallel) PAR=$2; shift 2;; --only) ONLY=$2; shift 2;; *) shift;; esac; done
M=data/manifests/terminal_bench_2_tasks.json
COMMIT=$(python3 -c "import json;print(json.load(open('$M'))['dataset_git_commit'][0])")
URL=$(python3 -c "import json;print(json.load(open('$M'))['dataset_git_url'][0])")
SRC=third_party/terminal-bench-2
if [ ! -d "$SRC/.git" ]; then git clone --quiet "$URL" "$SRC"; fi
git -C "$SRC" fetch --quiet origin "$COMMIT" && git -C "$SRC" checkout --quiet "$COMMIT"
TASKS=$(python3 -c "import json;print('\n'.join(json.load(open('$M'))['task_ids']))")
[ -n "$ONLY" ] && TASKS=$(echo "$ONLY" | tr ',' '\n')
mkdir -p logs/images
docker buildx inspect tb2builder >/dev/null 2>&1 || docker buildx create --name tb2builder --use >/dev/null
build_one() {
  t=$1; d="$SRC/$t/environment"
  if [ ! -f "$d/Dockerfile" ]; then echo "SKIP  $t (no environment/Dockerfile)"; echo "$t" >> logs/images/tb2_no_dockerfile.txt; return 0; fi
  if ls "$d"/docker-compose.y*ml >/dev/null 2>&1; then echo "COMPOSE $t (multi-container, handled separately)"; echo "$t" >> logs/images/tb2_compose.txt; fi
  img="$ACR_REGISTRY/$t"
  if docker buildx imagetools inspect "$img:${COMMIT:0:8}" >/dev/null 2>&1; then echo "skip  $t (already pushed)"; return 0; fi
  echo "build $t"
  docker buildx build --platform linux/amd64 --provenance=false \
    -t "$img:latest" -t "$img:${COMMIT:0:8}" --push "$d" > "logs/images/tb2_build_$t.log" 2>&1 \
    && echo "done  $t" || { echo "FAIL  $t (see logs/images/tb2_build_$t.log)"; echo "$t" >> logs/images/tb2_failed.txt; return 1; }
}
export -f build_one; export SRC ACR_REGISTRY COMMIT
echo "$TASKS" | xargs -P "$PAR" -I{} bash -c 'build_one {}' || true
uv run --no-sync python scripts/lock_images.py terminal_bench_2
