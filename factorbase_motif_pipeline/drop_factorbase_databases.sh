#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONNECTION_CONFIG="${SCRIPT_DIR}/config.tmp"

usage() {
    cat <<'EOF'
Usage: ./drop_factorbase_databases.sh [--yes] [--dry-run] [--keep-base] <config_path | db_name>

Drops the base database from `dbname` and the databases FactorBase creates from it:
  <dbname>_setup
  <dbname>_BN
  <dbname>_CT
  <dbname>_global_counts
  <dbname>_CT_cache

Options:
  --yes        Skip the confirmation prompt.
  --dry-run    Print the databases and SQL without executing anything.
  --keep-base  Keep the original `dbname` database and only drop the FactorBase-created ones.
  -h, --help   Show this help text.

If you provide a `.cfg` file, the script reads dbaddress/dbname/dbusername from it.
If you provide a database name such as `ali`, the script uses `config.tmp` for
the MySQL connection settings and drops:
  ali
  ali_setup
  ali_BN
  ali_CT
  ali_global_counts
  ali_CT_cache

Examples:
  ./drop_factorbase_databases.sh --dry-run proteins_experiment_config.cfg
  ./drop_factorbase_databases.sh --dry-run ali
EOF
}

get_config_value() {
    local key="$1"
    awk -v key="$key" '
        /^[[:space:]]*#/ { next }
        {
            raw_key = $0
            sub(/=.*/, "", raw_key)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", raw_key)

            if (raw_key == key) {
                value = substr($0, index($0, "=") + 1)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
                print value
                exit
            }
        }
    ' "$config_path"
}

confirm() {
    local reply
    read -r -p "Continue? [y/N] " reply
    [[ "$reply" =~ ^[Yy]([Ee][Ss])?$ ]]
}

confirm_delete_config() {
    local reply
    read -r -p "Delete config file '${config_path}' too? [y/N] " reply
    [[ "$reply" =~ ^[Yy]([Ee][Ss])?$ ]]
}

target_arg=""
config_path=""
delete_config_candidate=0
assume_yes=0
dry_run=0
drop_base=1
dbname=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes)
            assume_yes=1
            ;;
        --dry-run)
            dry_run=1
            ;;
        --keep-base)
            drop_base=0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -n "$target_arg" ]]; then
                echo "Only one config path or database name can be provided." >&2
                usage >&2
                exit 1
            fi
            target_arg="$1"
            ;;
    esac
    shift
done

if [[ -z "$target_arg" ]]; then
    echo "A config path or database name is required." >&2
    usage >&2
    exit 1
fi

if [[ "$target_arg" == *.cfg ]]; then
    config_path="$target_arg"
    delete_config_candidate=1
else
    config_path="$DEFAULT_CONNECTION_CONFIG"
    dbname="$target_arg"
fi

if [[ ! -f "$config_path" ]]; then
    echo "Config file not found: $config_path" >&2
    exit 1
fi

dbaddress="$(get_config_value "dbaddress")"
config_dbname="$(get_config_value "dbname")"
dbusername="$(get_config_value "dbusername")"
dbpassword="$(get_config_value "dbpassword")"

if [[ -z "$dbname" ]]; then
    dbname="$config_dbname"
fi

if [[ -z "$dbaddress" || -z "$dbname" || -z "$dbusername" ]]; then
    echo "The connection config must define dbaddress, dbname, and dbusername." >&2
    exit 1
fi

normalized_address="${dbaddress#mysql://}"
normalized_address="${normalized_address#mariadb://}"
normalized_address="${normalized_address%%/*}"

dbhost="${normalized_address%%:*}"
dbport="3306"
if [[ "$normalized_address" == *:* ]]; then
    dbport="${normalized_address##*:}"
fi

if [[ -z "$dbhost" ]]; then
    echo "Could not parse dbaddress: $dbaddress" >&2
    exit 1
fi

databases=()
if [[ "$drop_base" -eq 1 ]]; then
    databases+=("$dbname")
fi
databases+=(
    "${dbname}_setup"
    "${dbname}_BN"
    "${dbname}_CT"
    "${dbname}_global_counts"
    "${dbname}_CT_cache"
)

sql=""
for database in "${databases[@]}"; do
    sql+="DROP DATABASE IF EXISTS \`${database}\`;"$'\n'
done

echo "Target MySQL server: ${dbhost}:${dbport}"
echo "Databases to drop:"
for database in "${databases[@]}"; do
    echo "  - ${database}"
done
echo "Connection config used:"
echo "  - ${config_path}"
if [[ "$delete_config_candidate" -eq 1 ]]; then
    echo "Config file eligible for deletion:"
    echo "  - ${config_path}"
else
    echo "Config file deletion:"
    echo "  - not applicable (database name mode using config.tmp)"
fi

if [[ "$dry_run" -eq 1 ]]; then
    echo
    echo "SQL to execute:"
    printf '%s' "$sql"
    exit 0
fi

if [[ "$assume_yes" -ne 1 ]] && ! confirm; then
    echo "Aborted."
    exit 0
fi

delete_config=0
if [[ "$delete_config_candidate" -eq 0 ]]; then
    delete_config=0
elif [[ "$assume_yes" -eq 1 ]]; then
    delete_config=1
elif confirm_delete_config; then
    delete_config=1
fi

if command -v mysql >/dev/null 2>&1; then
    mysql_client=(mysql)
elif command -v mariadb >/dev/null 2>&1; then
    mysql_client=(mariadb)
else
    echo "Could not find a mysql or mariadb client in PATH." >&2
    exit 1
fi

mysql_command=(
    "${mysql_client[@]}"
    "--host=${dbhost}"
    "--port=${dbport}"
    "--protocol=TCP"
    "--user=${dbusername}"
    "-e"
    "$sql"
)

if [[ -n "$dbpassword" ]]; then
    MYSQL_PWD="$dbpassword" "${mysql_command[@]}"
else
    "${mysql_command[@]}"
fi

echo "Finished dropping ${#databases[@]} database(s)."
if [[ "$delete_config" -eq 1 ]]; then
    rm -f -- "$config_path"
    echo "Deleted config file: ${config_path}"
elif [[ "$delete_config_candidate" -eq 1 ]]; then
    echo "Kept config file: ${config_path}"
else
    echo "No config file was deleted."
fi
