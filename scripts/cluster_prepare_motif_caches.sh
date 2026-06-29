#!/usr/bin/env bash
set -u

SCHEDULE_FILE="CLUSTER_GPU_CONFIGS_SAMPLE.txt"
PYTHON_BIN="python"
MOTIF_CACHE_DIR="cache_motifs"
CACHE_ARCHIVE_ROOT="cache_motifs_archive"
MANIFEST_FILE=""
DRY_RUN=false

usage() {
  cat <<'EOF'
Rebuild local motif pickle caches for the configs in a schedule file.

This script is local-only: run it on the controller machine where
FactorBase/MySQL is available. It always starts from a fresh motif cache
directory, so old pickle files are not reused.

Usage:
  scripts/cluster_prepare_motif_caches.sh [options]

Options:
  --schedule FILE       Input file with rows: HOST GPU CONFIG_YAML
  --python-bin BIN      Python executable to run main.py
  --motif-cache-dir DIR Motif cache directory; default: cache_motifs
  --archive-root DIR    Where old motif cache dirs are moved before rebuild
  --manifest FILE       Manifest path; default: <motif-cache-dir>/MOTIF_CACHE_MANIFEST.tsv
  --dry-run             Print commands without running them
  --help                Show this help
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

config_uses_motif_loss() {
  local config_path="$1"
  grep -Eq '^[[:space:]]*motif_loss:[[:space:]]*true[[:space:]]*$' "$config_path"
}

clean_cache_dir() {
  if [[ -z "$MOTIF_CACHE_DIR" || "$MOTIF_CACHE_DIR" == "/" || "$MOTIF_CACHE_DIR" == "." || "$MOTIF_CACHE_DIR" == ".." ]]; then
    echo "Unsafe motif cache directory: $MOTIF_CACHE_DIR" >&2
    exit 2
  fi

  if [[ -e "$MOTIF_CACHE_DIR" ]]; then
    archive_name="$(basename "$MOTIF_CACHE_DIR")_$(date +%Y%m%d_%H%M%S)"
    archive_path="${CACHE_ARCHIVE_ROOT%/}/$archive_name"
    if [[ -e "$archive_path" ]]; then
      archive_path="${archive_path}_$$"
    fi

    run_cmd mkdir -p "$CACHE_ARCHIVE_ROOT"
    run_cmd mv -- "$MOTIF_CACHE_DIR" "$archive_path"
    echo "[motif-cache] archived previous cache: $archive_path"
  fi

  run_cmd mkdir -p "$MOTIF_CACHE_DIR"
}

write_manifest() {
  local pkl_path sha size mtime

  if [[ "$DRY_RUN" == true ]]; then
    echo "[manifest] dry-run target: $MANIFEST_FILE"
    return 0
  fi

  mkdir -p "$(dirname "$MANIFEST_FILE")"
  printf 'motif_pickle_path\tsha256\tsize_bytes\tmtime_epoch\n' > "$MANIFEST_FILE"

  while IFS= read -r pkl_path; do
    sha="$(sha256sum "$pkl_path" | awk '{print $1}')"
    size="$(wc -c < "$pkl_path" | tr -d ' ')"
    mtime="$(stat -c '%Y' "$pkl_path")"
    printf '%s\t%s\t%s\t%s\n' "$pkl_path" "$sha" "$size" "$mtime" >> "$MANIFEST_FILE"
  done < <(find "$MOTIF_CACHE_DIR" -maxdepth 1 -type f -name '*.pkl' -print | sort)
}

while (($#)); do
  case "$1" in
    --schedule)
      SCHEDULE_FILE="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --motif-cache-dir)
      MOTIF_CACHE_DIR="$2"
      shift 2
      ;;
    --archive-root)
      CACHE_ARCHIVE_ROOT="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST_FILE="$2"
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

if [[ ! -f "$SCHEDULE_FILE" ]]; then
  echo "Missing schedule file: $SCHEDULE_FILE" >&2
  exit 2
fi

if [[ -z "$MANIFEST_FILE" ]]; then
  MANIFEST_FILE="${MOTIF_CACHE_DIR%/}/MOTIF_CACHE_MANIFEST.tsv"
fi

seen_configs=" "
prepared=0
skipped=0
failures=0

echo "[motif-cache] fresh rebuild in $MOTIF_CACHE_DIR"
clean_cache_dir
echo "[motif-cache] reading $SCHEDULE_FILE"

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line%%#*}"
  line_is_blank "$line" && continue

  read -r host gpu config_path extra <<<"$line"
  if [[ -z "${host:-}" || -z "${gpu:-}" || -z "${config_path:-}" || -n "${extra:-}" ]]; then
    echo "[motif-cache] bad schedule row: $raw_line" >&2
    failures=$((failures + 1))
    continue
  fi

  if [[ "$seen_configs" == *" $config_path "* ]]; then
    continue
  fi
  seen_configs+=" $config_path "

  if [[ ! -f "$config_path" ]]; then
    echo "[motif-cache] missing config: $config_path" >&2
    failures=$((failures + 1))
    continue
  fi

  if ! config_uses_motif_loss "$config_path"; then
    echo "[motif-cache] skip non-motif config: $config_path"
    skipped=$((skipped + 1))
    continue
  fi

  echo "[motif-cache] prepare: $config_path"
  if run_cmd env MPLBACKEND=Agg PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u main.py \
    --config "$config_path" \
    --motif_cache_dir "$MOTIF_CACHE_DIR" \
    --prepare_motif_cache_only true; then
    prepared=$((prepared + 1))
  else
    echo "[motif-cache] failed: $config_path" >&2
    failures=$((failures + 1))
  fi
done < "$SCHEDULE_FILE"

write_manifest

echo
echo "[summary] prepared: $prepared"
echo "[summary] skipped non-motif: $skipped"
echo "[summary] manifest: $MANIFEST_FILE"
echo "[summary] failures: $failures"
exit "$failures"
