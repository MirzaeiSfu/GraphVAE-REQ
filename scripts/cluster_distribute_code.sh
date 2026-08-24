#!/usr/bin/env bash
set -euo pipefail

REPO_PATHS_FILE="CLUSTER_REPO_PATHS.txt"
CODE_SOURCE_DIR="."
SSH_CONNECT_TIMEOUT=10
DRY_RUN=false
SYNC_INPUTS=false
SYNC_PATHS=("data_raw" "cache_motifs")
CUSTOM_SYNC_PATHS=false
BO_CACHE_PATH=""
BO_CACHE_MANIFEST=""
PYTHON_BIN="${PYTHON_BIN:-python}"
REMOTE_PYTHON_BIN="${REMOTE_PYTHON_BIN:-python3}"
DEPLOYMENT_MANIFEST_TMP=""
SELECTED_HOST=""

usage() {
  cat <<'EOF'
Rsync controller code to each worker, optionally sync inputs.

Usage:
  scripts/cluster_distribute_code.sh [options]

Options:
  --repo-paths FILE          Input file with rows: HOST REPO_PATH
  --host HOST                Stage only the named host from --repo-paths
  --code-source DIR          Controller repo directory to sync; default: .
  --sync-inputs              Rsync data_raw and cache_motifs to workers.
                              When syncing cache_motifs, the remote cache_motifs
                              directory is removed first.
  --sync-path PATH           Extra/replacement path to sync; repeatable
  --bo-cache PATH            Stage this prebuilt BO dataset cache with checksums
  --bo-cache-manifest FILE   Manifest for --bo-cache; both options are required
  --local-python PATH        Python used to build the deployment manifest
  --remote-python PATH       Python used for remote manifest verification
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

check_clean_code_source() {
  local status_output
  local commit_id

  if ! git -C "$CODE_SOURCE_DIR_ABS" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Code source is not a git worktree: $CODE_SOURCE_DIR_ABS" >&2
    exit 2
  fi

  status_output="$(git -C "$CODE_SOURCE_DIR_ABS" status --short)"
  if [[ -n "$status_output" ]]; then
    if [[ "$DRY_RUN" == true ]]; then
      echo "[code] dirty git worktree; real distribute would fail:"
      printf '%s\n' "$status_output" | sed 's/^/[code]   /'
      return 0
    fi

    echo "Refusing to distribute uncommitted code from: $CODE_SOURCE_DIR_ABS" >&2
    printf '%s\n' "$status_output" | sed 's/^/  /' >&2
    echo "Commit, stash, or remove these changes before running distribute." >&2
    exit 2
  fi

  commit_id="$(git -C "$CODE_SOURCE_DIR_ABS" rev-parse --short HEAD)"
  echo "[code] clean git worktree: $commit_id"
}

while (($#)); do
  case "$1" in
    --repo-paths)
      REPO_PATHS_FILE="$2"
      shift 2
      ;;
    --code-source)
      CODE_SOURCE_DIR="$2"
      shift 2
      ;;
    --host)
      SELECTED_HOST="$2"
      shift 2
      ;;
    --remote-url)
      echo "--remote-url is ignored; code is synced from the controller with rsync." >&2
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
    --bo-cache)
      BO_CACHE_PATH="$2"
      shift 2
      ;;
    --bo-cache-manifest)
      BO_CACHE_MANIFEST="$2"
      shift 2
      ;;
    --local-python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --remote-python)
      REMOTE_PYTHON_BIN="$2"
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

if ! CODE_SOURCE_DIR_ABS="$(cd "$CODE_SOURCE_DIR" && pwd)"; then
  echo "Missing code source directory: $CODE_SOURCE_DIR" >&2
  exit 2
fi

check_clean_code_source

if [[ -n "$BO_CACHE_PATH" || -n "$BO_CACHE_MANIFEST" ]]; then
  if [[ -z "$BO_CACHE_PATH" || -z "$BO_CACHE_MANIFEST" ]]; then
    echo "--bo-cache and --bo-cache-manifest must be supplied together." >&2
    exit 2
  fi
  if [[ ! -f "$BO_CACHE_PATH" || ! -f "$BO_CACHE_MANIFEST" ]]; then
    echo "BO cache or cache manifest is missing." >&2
    exit 2
  fi
  if ! cache_contract="$($PYTHON_BIN -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); print(value["sha256"]); print(value["relative_path"])' "$BO_CACHE_MANIFEST")"; then
    echo "Local Python could not read the BO cache manifest: $PYTHON_BIN" >&2
    exit 2
  fi
  expected_cache_sha="${cache_contract%%$'\n'*}"
  cache_relative_path="${cache_contract#*$'\n'}"
  if [[ "$cache_relative_path" == /* || "$cache_relative_path" == ".." || "$cache_relative_path" == ../* || "$cache_relative_path" == */../* ]]; then
    echo "BO cache manifest relative_path is unsafe: $cache_relative_path" >&2
    exit 2
  fi
  actual_cache_sha="$(sha256sum "$BO_CACHE_PATH" | awk '{print $1}')"
  if [[ "$expected_cache_sha" != "$actual_cache_sha" ]]; then
    echo "BO cache SHA-256 differs from its manifest." >&2
    exit 2
  fi
fi

DEPLOYMENT_MANIFEST_TMP="$(mktemp "${TMPDIR:-/tmp}/graphvae-bo-deployment.XXXXXX.json")"
trap '[[ -n "$DEPLOYMENT_MANIFEST_TMP" ]] && rm -f "$DEPLOYMENT_MANIFEST_TMP"' EXIT
if ! run_cmd "$PYTHON_BIN" "$CODE_SOURCE_DIR_ABS/scripts/graphvae_attr_bo_fingerprints.py" \
  --deployment-root "$CODE_SOURCE_DIR_ABS" --output "$DEPLOYMENT_MANIFEST_TMP"; then
  echo "Local deployment manifest generation failed; nothing was transferred." >&2
  exit 2
fi
if [[ "$DRY_RUN" == false && ! -s "$DEPLOYMENT_MANIFEST_TMP" ]]; then
  echo "Local deployment manifest is empty; nothing was transferred." >&2
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
matched_hosts=0
SSH_OPTS=(-o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" -o StrictHostKeyChecking=accept-new)
RSYNC_SSH_CMD="ssh -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o StrictHostKeyChecking=accept-new"
CODE_RSYNC_EXCLUDES=(
  --exclude .git/
  --exclude .graphcl_deps/
  --exclude .runtime/
  --exclude __pycache__/
  --exclude .pytest_cache/
  --exclude cache_datasets/
  --exclude cache_motifs/
  --exclude cache_motifs_archive/
  --exclude collected_runs/
  --exclude data_raw/
  --exclude graph_evaluation_inputs/
  --exclude runs/
)

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line%%#*}"
  line_is_blank "$line" && continue

  read -r host repo_path extra <<<"$line"
  if [[ -z "${host:-}" || -z "${repo_path:-}" || -n "${extra:-}" ]]; then
    echo "[repo] bad repo path row: $raw_line" >&2
    failures=$((failures + 1))
    continue
  fi
  if [[ -n "$SELECTED_HOST" && "$host" != "$SELECTED_HOST" ]]; then
    continue
  fi
  matched_hosts=$((matched_hosts + 1))

  echo
  echo "[code] $CODE_SOURCE_DIR_ABS -> $host:$repo_path"

  parent_dir="$(dirname "$repo_path")"
  repo_q="$(quote_one "$repo_path")"
  parent_q="$(quote_one "$parent_dir")"

  remote_script=$(cat <<EOF
set -euo pipefail
mkdir -p $parent_q
mkdir -p $repo_q
EOF
)

  ssh_cmd=(ssh -n "${SSH_OPTS[@]}" "$host" "bash -lc $(quote_one "$remote_script")")
  if ! run_cmd "${ssh_cmd[@]}"; then
    echo "[code] failed to create repo directory on $host; continuing" >&2
    failures=$((failures + 1))
    continue
  fi

  code_rsync_args=(-az --delete "${CODE_RSYNC_EXCLUDES[@]}")
  if ! run_cmd rsync "${code_rsync_args[@]}" -e "$RSYNC_SSH_CMD" "$CODE_SOURCE_DIR_ABS/" "$host:$repo_path/"; then
    echo "[code] failed on $host; continuing" >&2
    failures=$((failures + 1))
    continue
  fi

  if ! run_cmd rsync -azc -e "$RSYNC_SSH_CMD" "$DEPLOYMENT_MANIFEST_TMP" "$host:$repo_path/deployment_manifest.json"; then
    echo "[code] deployment manifest sync failed on $host" >&2
    failures=$((failures + 1))
    continue
  fi
  remote_verify_script="set -euo pipefail; $(quote_one "$REMOTE_PYTHON_BIN") $(quote_one "$repo_path/scripts/graphvae_attr_bo_fingerprints.py") --deployment-root $(quote_one "$repo_path") --verify-manifest $(quote_one "$repo_path/deployment_manifest.json")"
  verify_cmd=(ssh -n "${SSH_OPTS[@]}" "$host" "bash -lc $(quote_one "$remote_verify_script")")
  if ! run_cmd "${verify_cmd[@]}"; then
    echo "[code] remote deployment hash verification failed on $host" >&2
    failures=$((failures + 1))
    continue
  fi

  if [[ -n "$BO_CACHE_PATH" ]]; then
    remote_cache_path="${repo_path%/}/$cache_relative_path"
    remote_cache_dir="$(dirname "$remote_cache_path")"
    cache_name="$(basename "$remote_cache_path")"
    if ! run_cmd ssh -n "${SSH_OPTS[@]}" "$host" mkdir -p "$remote_cache_dir"; then
      failures=$((failures + 1))
      continue
    fi
    if ! run_cmd rsync -azc -e "$RSYNC_SSH_CMD" "$BO_CACHE_PATH" "$host:$remote_cache_dir/$cache_name"; then
      failures=$((failures + 1))
      continue
    fi
    if ! run_cmd rsync -azc -e "$RSYNC_SSH_CMD" "$BO_CACHE_MANIFEST" "$host:$repo_path/dataset_cache_manifest.json"; then
      failures=$((failures + 1))
      continue
    fi
    remote_cache_verify_cmd="set -euo pipefail; $(quote_one "$REMOTE_PYTHON_BIN") $(quote_one "$repo_path/scripts/prepare_graphvae_attr_bo_cache.py") --cache-path $(quote_one "$remote_cache_dir/$cache_name") --verify-manifest $(quote_one "$repo_path/dataset_cache_manifest.json") --make-read-only"
    if ! run_cmd ssh -n "${SSH_OPTS[@]}" "$host" "bash -lc $(quote_one "$remote_cache_verify_cmd")"; then
      echo "[cache] remote cache/split/schema verification failed on $host" >&2
      failures=$((failures + 1))
      continue
    fi
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
        clean_cmd=(ssh -n "${SSH_OPTS[@]}" "$host" "bash -lc $(quote_one "$remote_clean_script")")
        if ! run_cmd "${clean_cmd[@]}"; then
          echo "[sync] failed to clear remote cache on $host; continuing" >&2
          failures=$((failures + 1))
          continue
        fi
      fi

      echo "[sync] $sync_path -> $host:$repo_path/"
      if ! run_cmd rsync "${rsync_args[@]}" -e "$RSYNC_SSH_CMD" "$sync_path" "$host:$repo_path/"; then
        echo "[sync] failed on $host for $sync_path; continuing" >&2
        failures=$((failures + 1))
      fi
    done
  fi
done < "$REPO_PATHS_FILE"

if [[ -n "$SELECTED_HOST" && "$matched_hosts" -eq 0 ]]; then
  echo "No repo-path entry found for selected host: $SELECTED_HOST" >&2
  exit 2
fi

echo
echo "[summary] failures: $failures"
exit "$failures"
