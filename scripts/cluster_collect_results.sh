#!/usr/bin/env bash
set -u

REPO_PATHS_FILE="CLUSTER_REPO_PATHS.txt"
REMOTE_RUN_ROOT="runs/distributed"
COLLECT_ROOT="collected_runs"
DATE_PREFIX="$(date +%Y%m%d)"
SSH_CONNECT_TIMEOUT=10
DRY_RUN=false

usage() {
  cat <<'EOF'
Collect distributed run outputs back to this machine.

Usage:
  scripts/cluster_collect_results.sh [options]

Options:
  --repo-paths FILE          Input file with rows: HOST REPO_PATH
  --remote-run-root PATH     Remote runs folder inside each repo
  --collect-root PATH        Local collection folder
  --date-prefix YYYYMMDD     Local collection batch folder; default is today
  --ssh-connect-timeout SEC  SSH connection timeout
  --dry-run                  Print commands without running them
  --help                     Show this help
EOF
}

quote_words() {
  local quoted=()
  local word
  local word_q
  for word in "$@"; do
    printf -v word_q '%q' "$word"
    quoted+=("$word_q")
  done
  printf '%s' "${quoted[*]}"
}

run_cmd() {
  echo "+ $(quote_words "$@")"
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi
  "$@"
}

line_is_blank() {
  [[ -z "${1//[[:space:]]/}" ]]
}

while (($#)); do
  case "$1" in
    --repo-paths)
      REPO_PATHS_FILE="$2"
      shift 2
      ;;
    --remote-run-root)
      REMOTE_RUN_ROOT="$2"
      shift 2
      ;;
    --collect-root)
      COLLECT_ROOT="$2"
      shift 2
      ;;
    --date-prefix)
      DATE_PREFIX="$2"
      shift 2
      ;;
    --ssh-connect-timeout)
      SSH_CONNECT_TIMEOUT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$REPO_PATHS_FILE" ]]; then
  echo "Missing repo paths file: $REPO_PATHS_FILE" >&2
  exit 2
fi

failures=0
RSYNC_SSH_CMD="ssh -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o StrictHostKeyChecking=accept-new"

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line%%#*}"
  line_is_blank "$line" && continue

  read -r host repo_path extra <<<"$line"
  if [[ -z "${host:-}" || -z "${repo_path:-}" || -n "${extra:-}" ]]; then
    echo "[collect] bad repo path row: $raw_line" >&2
    failures=$((failures + 1))
    continue
  fi

  local_dest="$COLLECT_ROOT/$DATE_PREFIX/$host"
  remote_source="$host:${repo_path%/}/${REMOTE_RUN_ROOT%/}/"

  echo
  echo "[collect] $remote_source -> $local_dest/"
  if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$local_dest"
  fi

  if ! run_cmd rsync -az -e "$RSYNC_SSH_CMD" "$remote_source" "$local_dest/"; then
    echo "[collect] failed on $host; continuing" >&2
    failures=$((failures + 1))
  fi
done < "$REPO_PATHS_FILE"

echo
echo "[summary] failures: $failures"
exit "$failures"
