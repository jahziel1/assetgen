#!/bin/bash
# uso: gpurun.sh <etiqueta> <comando...>  -> flock + tiempo + VRAM pico (nvidia-smi) en /root/stats/<etiqueta>.json
tag=$1; shift
mkdir -p /root/stats
flock /root/gpu.lock bash -c '
tag='"$tag"'
base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
( peak=0; while true; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$m" -gt "$peak" ] && peak=$m; echo $peak > /root/stats/$tag.peak; sleep 1; done ) &
SP=$!
s=$(date +%s.%N)
"$@"
rc=$?
e=$(date +%s.%N)
kill $SP 2>/dev/null
peak=$(cat /root/stats/$tag.peak)
python3 -c "import json;print(json.dumps({\"stage\":\"$tag\",\"seconds\":round($e-$s,1),\"vram_peak_mib\":$peak,\"vram_base_mib\":$base,\"rc\":$rc}))" > /root/stats/$tag.json
cat /root/stats/$tag.json
exit $rc
' _ "$@"
