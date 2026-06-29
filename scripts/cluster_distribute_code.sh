#!/usr/bin/env bash
set -u

REPO_PATHS_FILE="CLUSTER_REPO_PATHS.txt"
REMOTE_URL="git@github.com:MirzaeiSfu/GraphVAE-REQ.git"
SSH_CONNECT_TIMEOUT=10
DRY_RUN=false
SYNC_INPUTS=false
SYNC_PATHS=("data_raw" "cache_motifs")
CUSTOM_SYNC_PATHS=false

usage() {
  cat <<'EOF'
Clone/pull the repo on each worker, optionally sync inputs.

Usage:
  scripts/cluster_distribute_code.sh [options]

Options:
  --repo-paths FILE          Input file with rows: HOST REPO_PATH
  --remote-url URL           Git remote for missing worker repos
  --sync-inputs              Rsync data_raw and cache_motifs to workers.
                              When syncing cache_motifs, the remote cache_motifs
                              directory is removed first.
  --sync-path PATH           Extra/replacement path to sync; repeatable
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

quote_one() {
  printf '%q' "$1"
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
    --remote-url)
      REMOTE_URL="$2"
      shift 2
      ;;
    --sync-inputs)
      SYNC_INPUTS=true
      shift
      ;;
    --sync-path)
      if [[ "$CUSTOM_SYNC_PATHS" == false ]]; then
        SYNC_PATHS=()
        CUSTOM_SYNC_PATHS=true
      fi
      SYNC_PATHS+=("$2")
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

if [[ "$SYNC_INPUTS" == true ]]; then
  for sync_path in "${SYNC_PATHS[@]}"; do
    if [[ ! -e "$sync_path" ]]; then
      echo "Missing sync path: $sync_path" >&2
      echo "Create it first, or pass explicit --sync-path entries." >&2
      exit 2
    fi
    if [[ "$sync_path" == "cache_motifs" ]]; then
      first_pickle="$(find "$sync_path" -maxdepth 1 -type f -name '*.pkl' -print -quit)"
      if [[ -z "$first_pickle" ]]; then
        echo "No motif pickle files found in $sync_path" >&2
        echo "Run scripts/cluster_prepare_motif_caches.sh before syncing cache_motifs." >&2
        exit 2
      fi
    fi
  done
fi

failures=0

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line%%#*}"
  line_is_blank "$line" && continue

  read -r host repo_path extra <<<"$line"
  if [[ -z "${host:-}" || -z "${repo_path:-}" || -n "${extra:-}" ]]; then
    echo "[repo] bad repo path row: $raw_line" >&2
    failures=$((failures + 1))
    continue
  fi

  echo
  echo "[repo] $host -> $repo_path"

  parent_dir="$(dirname "$repo_path")"
  repo_q="$(quote_one "$repo_path")"
  repo_git_q="$(quote_one "$repo_path/.git")"
  parent_q="$(quote_one "$parent_dir")"
  remote_url_q="$(quote_one "$REMOTE_URL")"

  remote_script=$(cat <<EOF
set -euo pipefail
if [[ -d $repo_git_q ]]; then
  cd $repo_q
  git pull --ff-only
else
  mkdir -p $parent_q
  git clone $remote_url_q $repo_q
fi
EOF
)

  ssh_cmd=(ssh -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" "$host" "bash -lc $(quote_one "$remote_script")")
  if ! run_cmd "${ssh_cmd[@]}"; then
    echo "[repo] failed on $host; continuing" >&2
    failures=$((failures + 1))
    continue
  fi

  if [[ "$SYNC_INPUTS" == true ]]; then
    for sync_path in "${SYNC_PATHS[@]}"; do
      rsync_args=(-az)
      if [[ "$sync_path" == "cache_motifs" || "$sync_path" == cache_motifs/* ]]; then
        rsync_args=(-azc)
      fi

      if [[ "$sync_path" == "cache_motifs" ]]; then
        remote_cache_path="${repo_path%/}/cache_motifs"
        remote_cache_q="$(quote_one "$remote_cache_path")"
        remote_clean_script=$(cat <<EOF
set -euo pipefail
rm -rf $remote_cache_q
mkdir -p $remote_cache_q
EOF
)
        echo "[sync] clear remote cache: $host:$remote_cache_path"
        clean_cmd=(ssh -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" "$host" "bash -lc $(quote_one "$remote_clean_script")")
        if ! run_cmd "${clean_cmd[@]}"; then
          echo "[sync] failed to clear remote cache on $host; continuing" >&2
          failures=$((failures + 1))
          continue
        fi
      fi

      echo "[sync] $sync_path -> $host:$repo_path/"
      if ! run_cmd rsync "${rsync_args[@]}" "$sync_path" "$host:$repo_path/"; then
        echo "[sync] failed on $host for $sync_path; continuing" >&2
        failures=$((failures + 1))
      fi
    done
  fi
done < "$REPO_PATHS_FILE"

echo
echo "[summary] failures: $failures"
exit "$failures"
