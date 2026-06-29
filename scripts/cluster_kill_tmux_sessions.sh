#!/usr/bin/env bash
set -u

REPO_PATHS_FILE="CLUSTER_REPO_PATHS.txt"
SSH_CONNECT_TIMEOUT=10
KILL_SESSIONS=false
MATCH_REGEX=""

usage() {
  cat <<'EOF'
List or kill tmux sessions on every host in CLUSTER_REPO_PATHS.txt.

By default this is a dry run: it prints the sessions that would be killed.
Pass --yes to actually kill them.

Usage:
  scripts/cluster_kill_tmux_sessions.sh
  scripts/cluster_kill_tmux_sessions.sh --yes
  scripts/cluster_kill_tmux_sessions.sh --match '^grid_2epoch_' --yes

Options:
  --repo-paths FILE          Input file with rows: HOST REPO_PATH
  --match REGEX             Only target tmux session names matching REGEX
  --ssh-connect-timeout SEC  SSH connection timeout
  --yes                     Actually kill matching sessions
  --help                    Show this help
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

line_is_blank() {
  [[ -z "${1//[[:space:]]/}" ]]
}

run_cmd() {
  echo "+ $(quote_words "$@")"
  "$@"
}

while (($#)); do
  case "$1" in
    --repo-paths)
      REPO_PATHS_FILE="$2"
      shift 2
      ;;
    --match)
      MATCH_REGEX="$2"
      shift 2
      ;;
    --ssh-connect-timeout)
      SSH_CONNECT_TIMEOUT="$2"
      shift 2
      ;;
    --yes)
      KILL_SESSIONS=true
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

if [[ "$KILL_SESSIONS" != true ]]; then
  echo "[dry-run] no sessions will be killed; pass --yes to kill them"
fi

failures=0
SSH_OPTS=(-n -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" -o StrictHostKeyChecking=accept-new)

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line%%#*}"
  line_is_blank "$line" && continue

  read -r host repo_path extra <<<"$line"
  if [[ -z "${host:-}" || -z "${repo_path:-}" || -n "${extra:-}" ]]; then
    echo "[tmux] bad repo path row: $raw_line" >&2
    failures=$((failures + 1))
    continue
  fi

  echo
  echo "[tmux] $host"

  match_q="$(quote_one "$MATCH_REGEX")"
  kill_q="$(quote_one "$KILL_SESSIONS")"
  remote_script=$(cat <<EOF
set -u
match_regex=$match_q
kill_sessions=$kill_q

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed on this host."
  exit 0
fi

sessions="\$(tmux list-sessions -F '#S' 2>/dev/null || true)"
if [[ -z "\$sessions" ]]; then
  echo "no tmux sessions"
  exit 0
fi

if [[ -n "\$match_regex" ]]; then
  sessions="\$(printf '%s\n' "\$sessions" | grep -E -- "\$match_regex" || true)"
fi

if [[ -z "\$sessions" ]]; then
  echo "no matching tmux sessions"
  exit 0
fi

if [[ "\$kill_sessions" != true ]]; then
  printf '%s\n' "\$sessions" | sed 's/^/[dry-run] would kill: /'
  exit 0
fi

while IFS= read -r session_name; do
  [[ -z "\$session_name" ]] && continue
  tmux kill-session -t "\$session_name"
  echo "[kill] \$session_name"
done <<< "\$sessions"
EOF
)

  ssh_cmd=(ssh "${SSH_OPTS[@]}" "$host" "bash -lc $(quote_one "$remote_script")")
  if ! run_cmd "${ssh_cmd[@]}"; then
    echo "[tmux] failed on $host; continuing" >&2
    failures=$((failures + 1))
  fi
done < "$REPO_PATHS_FILE"

echo
echo "[summary] failures: $failures"
exit "$failures"
