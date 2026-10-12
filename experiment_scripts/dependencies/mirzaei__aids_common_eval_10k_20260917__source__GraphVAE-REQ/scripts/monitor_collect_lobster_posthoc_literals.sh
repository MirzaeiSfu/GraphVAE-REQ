#!/usr/bin/env bash
set -euo pipefail

DATE_PREFIX="${DATE_PREFIX:-20260716}"
SCHEDULE="${SCHEDULE:-CLUSTER_GPU_CONFIGS_LOBSTER_POSTHOC_LITERALS.txt}"
REPO_PATHS="${REPO_PATHS:-CLUSTER_REPO_PATHS_LOBSTER_POSTHOC_LITERALS.txt}"
RUN_ROOT="${RUN_ROOT:-runs/20260716/lobster_posthoc_literals}"
COLLECT_ROOT="${COLLECT_ROOT:-collected_runs}"
POLL_SECONDS="${POLL_SECONDS:-30}"

while true; do
  active=0
  complete=0
  failed=0
  while read -r host gpu config extra; do
    [[ -z "${host:-}" || "$host" == \#* ]] && continue
    repo="$(awk -v host="$host" '$1 == host { print $2; exit }' "$REPO_PATHS")"
    name="$(basename "$config" .yaml)"
    session="${DATE_PREFIX}_${name}__${host}_gpu${gpu}"
    run_dir="$repo/$RUN_ROOT/${name}__${host}_gpu${gpu}"
    state="$(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$host" \
      "if tmux has-session -t '$session' 2>/dev/null; then echo active; \
       elif grep -q 'trainning time:' '$run_dir/stdout.log' 2>/dev/null; then echo complete; \
       else echo failed; fi")"
    case "$state" in
      active) active=$((active + 1)) ;;
      complete) complete=$((complete + 1)) ;;
      *)
        echo "[monitor] failed: $host gpu$gpu $config" >&2
        failed=$((failed + 1))
        ;;
    esac
  done < "$SCHEDULE"

  echo "[monitor] active=$active complete=$complete failed=$failed"
  ((failed == 0)) || exit 1
  ((active > 0)) || break
  sleep "$POLL_SECONDS"
done

scripts/cluster_collect_results.sh \
  --repo-paths "$REPO_PATHS" \
  --remote-run-root "$RUN_ROOT" \
  --collect-root "$COLLECT_ROOT" \
  --date-prefix "$DATE_PREFIX"

echo "[monitor] collected into $COLLECT_ROOT/$DATE_PREFIX/$(basename "$RUN_ROOT")"
