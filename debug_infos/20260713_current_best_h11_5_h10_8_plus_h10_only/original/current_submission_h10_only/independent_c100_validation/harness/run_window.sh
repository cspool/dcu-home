#!/usr/bin/env bash
# Repeated fixed all3 bodies until the natural active window reaches 600 s.

set -euo pipefail

ROOT=/public/home/tangyu408/testdata/goal_runs/20260712_r27_p9_c100_fixed_all3_600
SERVICE=$ROOT/service
OUT=$ROOT/all3_natural_600
SCRIPT=/public/home/tangyu408/testdata/run_throughput.sh
SCRIPT_SHA=adf0cf91266745b37df916926c7d495ec79f00a11be653c219d1d5df4d93c681
MINIMUM_SECONDS=600

die() { echo "ERROR: $*" >&2; exit 1; }
[[ ${P9_ALLOW_C100_PERFORMANCE-} == YES_600S_AUTHORIZED ]] || die "missing C100 performance authorization"
[[ ! -e "$OUT" ]] || die "refusing reused performance window: $OUT"
python3 - "$SERVICE/service_state.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"]=="ready" and p["state"]=="C100"
assert p["wheel_kind"]=="candidate"
assert p["features"]=={"h10_confirmed":1,"s32":0,"hg3":0}
assert p["purpose"]=="fixed_all3_screen_performance"
assert p["tunable_controls"]=={"verbose":"unset","tuning":0,"record_untuned":0}
PY
pid=$(<"$SERVICE/service.pid")
pgid=$(<"$SERVICE/service.pgid")
[[ $pid =~ ^[1-9][0-9]*$ && $pgid == "$pid" ]] || die "invalid owned service identity"
kill -0 "$pid" 2>/dev/null || die "owned service is not live"
[[ $(ps -o pgid= -p "$pid" | tr -d ' ') == "$pgid" ]] || die "owned PGID drift"
curl --noproxy '*' --silent --show-error --fail --max-time 10 \
  http://127.0.0.1:8001/v1/models >/dev/null
echo "$SCRIPT_SHA  $SCRIPT" | sha256sum -c - >/dev/null

mkdir -p "$OUT"
start_ns=$(date +%s%N)
round=0
printf 'round\tstart_ns\tend_ns\telapsed_ns\tresult_root\tscript_sha_before\tscript_sha_after\n' >"$OUT/rounds.tsv"
while :; do
  round=$((round + 1))
  round_dir=$(printf '%s/round-%04d' "$OUT" "$round")
  [[ ! -e "$round_dir" ]] || die "round collision: $round_dir"
  mkdir -p "$round_dir"
  before=$(sha256sum "$SCRIPT" | awk '{print $1}')
  [[ $before == "$SCRIPT_SHA" ]] || die "fixed script changed before round $round"
  round_start=$(date +%s%N)
  echo "C100 natural window round $round started"
  (
    cd /public/home/tangyu408/testdata
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u MODEL_DIR -u SERVED_MODEL_NAME -u VLLM_HOST -u VLLM_PORT \
      -u MAX_CONCURRENCY -u REQUEST_RATE -u CUSTOM_OUTPUT_LEN -u NUM_WARMUPS \
      NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
      RESULT_ROOT="$round_dir/results" \
      ./run_throughput.sh all 3
  ) >"$round_dir/run_throughput.log" 2>&1
  round_end=$(date +%s%N)
  after=$(sha256sum "$SCRIPT" | awk '{print $1}')
  [[ $after == "$SCRIPT_SHA" ]] || die "fixed script changed after round $round"
  [[ $(find "$round_dir/results" -type f -name result.json | wc -l) -eq 3 ]] || die "round $round lacks three results"
  python3 - "$round_dir/results" <<'PY'
import json,sys
from pathlib import Path
paths=sorted(Path(sys.argv[1]).glob("*_throughput/result.json"))
assert len(paths)==3
for path in paths:
    p=json.load(open(path))
    assert p["num_prompts"]==3 and p["completed"]==3 and p["failed"]==0
    assert len(p["generated_texts"])==3 and len(p["output_lens"])==3
PY
  printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$round" "$round_start" "$round_end" "$((round_end-round_start))" \
    "$round_dir/results" "$before" "$after" >>"$OUT/rounds.tsv"
  now_ns=$(date +%s%N)
  echo "C100 natural window round $round complete; active_ns=$((now_ns-start_ns))"
  (( now_ns - start_ns >= MINIMUM_SECONDS * 1000000000 )) && break
done
end_ns=$(date +%s%N)
python3 - "$OUT/window.json" "$start_ns" "$end_ns" "$round" "$SCRIPT_SHA" <<'PY'
import json,sys
path,start,end,rounds,sha=sys.argv[1:]
p={
 "schema":"r27-h10.18-all3-natural-window-v1","status":"complete",
 "candidate_state":"C100","body_command":"./run_throughput.sh all 3",
 "start_ns":int(start),"end_ns":int(end),"elapsed_ns":int(end)-int(start),
 "minimum_seconds":600,"rounds_completed":int(rounds),
 "idle_padding_seconds":0,"natural_active_repetition":True,
 "fixed_script_sha256_before_after":sha,
}
with open(path,"w",encoding="utf-8") as f: json.dump(p,f,indent=2,sort_keys=True); f.write("\n")
PY
(( end_ns - start_ns >= MINIMUM_SECONDS * 1000000000 )) || die "natural window is below 600 seconds"
kill -0 "$pid" 2>/dev/null || die "owned service died during window"
echo "$SCRIPT_SHA  $SCRIPT" | sha256sum -c - >/dev/null
echo "C100 natural active window complete: rounds=$round elapsed_ns=$((end_ns-start_ns))"
