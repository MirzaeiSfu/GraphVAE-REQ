#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" != "3" ]]; then
    echo "Usage: $0 SOURCE_DATABASE DESTINATION_DATABASE CONFIG_FILE" >&2
    exit 2
fi

source_database="$1"
destination_database="$2"
factorbase_config="$3"
run_root="/localhome/mirzaei/fb_multi"
factorbase_jar="$run_root/factorbase-transitive-1.0-SNAPSHOT.jar"
run_log="$run_root/$destination_database.log"

mkdir -p "$run_root/work"
exec > >(tee -a "$run_log") 2>&1

database_user=$(awk -F= '/^[[:space:]]*dbusername[[:space:]]*=/{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}' "$factorbase_config")
database_password=$(awk -F= '/^[[:space:]]*dbpassword[[:space:]]*=/{sub(/^[^=]*=/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print}' "$factorbase_config")
export MYSQL_PWD="$database_password"

source_exists=$(mysql -h 127.0.0.1 -P 3306 -u "$database_user" -N -e \
    "SELECT COUNT(*) FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = '$source_database';")
if [[ "$source_exists" != "1" ]]; then
    echo "Source database does not exist: $source_database" >&2
    exit 1
fi

destination_exists=$(mysql -h 127.0.0.1 -P 3306 -u "$database_user" -N -e \
    "SELECT COUNT(*) FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = '$destination_database';")

if [[ "$destination_exists" == "0" ]]; then
    database_charset=$(mysql -h 127.0.0.1 -P 3306 -u "$database_user" -N -e \
        "SELECT DEFAULT_CHARACTER_SET_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = '$source_database';")
    database_collation=$(mysql -h 127.0.0.1 -P 3306 -u "$database_user" -N -e \
        "SELECT DEFAULT_COLLATION_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = '$source_database';")

    echo "Creating database copy: $source_database -> $destination_database"
    mysql -h 127.0.0.1 -P 3306 -u "$database_user" -e \
        "CREATE DATABASE \`$destination_database\` CHARACTER SET $database_charset COLLATE $database_collation;"

    mysqldump \
        -h 127.0.0.1 \
        -P 3306 \
        -u "$database_user" \
        --single-transaction \
        --routines \
        --triggers \
        --events \
        --no-tablespaces \
        "$source_database" | mysql -h 127.0.0.1 -P 3306 -u "$database_user" "$destination_database"
else
    echo "Using existing destination database: $destination_database"
fi

echo "Excluding run_metadata from FactorBase input if present"
mysql -h 127.0.0.1 -P 3306 -u "$database_user" -e \
    "DROP TABLE IF EXISTS \`$destination_database\`.\`run_metadata\`;"

echo "Starting FactorBase for $destination_database on $(hostname)"
cd "$run_root/work"
java -Dconfig="$factorbase_config" -jar "$factorbase_jar"
