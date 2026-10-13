#!/usr/bin/env bash
set -u
root=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921
exec >> "$root/legacy_fb_transfer.log" 2>&1
for host in 13 16 17 18 19; do
  base=/local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs
  if [ "$host" = 18 ] || [ "$host" = 19 ]; then base=/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs; fi
  while IFS= read -r source; do
    [ -z "$source" ] && continue
    ds=$(basename "$source" | tr '[:lower:]' '[:upper:]')
    case "$ds" in PTC|MUTAG|QM9|AIDS|OGB) ;; *) continue;; esac
    relative=${source#"$base/"}
    dest=$root/$ds/sources/legacy_fb_cs-cl-$host/$relative
    mkdir -p "$dest"
    rsync -a --partial "mirzaei@cs-cl-$host.cmpt.sfu.ca:$source/" "$dest/" || echo "FAILED $host $source"
  done < <(ssh "mirzaei@cs-cl-$host.cmpt.sfu.ca" "find $base -maxdepth 3 -type d \\( -iname '*ptc*' -o -iname '*mutag*' -o -iname '*qm9*' -o -iname '*aids*' -o -iname '*ogb*' \\) 2>/dev/null")
done
date -Is
