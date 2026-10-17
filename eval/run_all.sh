#!/usr/bin/env bash
# Run everything matrix.yaml assigns to one model.  usage: eval/run_all.sh <model>
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1
read -r GROUP FULL PROXY <<<"$(python3 - "$MODEL" <<'PY'
import sys, yaml
m = yaml.safe_load(open("eval/matrix.yaml"))["models"][sys.argv[1]]
print(m["group"], ",".join(m.get("full", [])) or "-", "1" if m.get("proxy") else "0")
PY
)"
echo "model=$MODEL group=$GROUP full=$FULL proxy=$PROXY"
[ "$PROXY" = "1" ] && eval/run_proxies.sh "$MODEL" "$GROUP"
for b in ${FULL//,/ }; do
  case $b in
    swebench_pro_v2_hard) eval/run_swebench_pro.sh "$MODEL" "$GROUP";;
    terminal_bench_2)     eval/run_terminal_bench.sh "$MODEL" "$GROUP";;
    hle)                  eval/run_hle.sh "$MODEL" "$GROUP";;
    asi_bench)            eval/run_asi_bench.sh "$MODEL" "$GROUP";;
    -) ;;
  esac
done
eval/collect.sh
