#!/usr/bin/env bash
set -u

show_host() {
    local host=$1 repo=$2
    echo "$host"
    ssh -o BatchMode=yes "mirzaei@${host}.cmpt.sfu.ca" "cd '$repo' && for f in run_status/motif_false_3seed/gv_*_false_s*.status; do [ -f \"\$f\" ] || continue; id=\$(basename \"\$f\" .status); state=\$(awk '{print \$1}' \"\$f\"); progress=\$(rg 'Epoch:' run_logs/motif_false_3seed/\${id}.log 2>/dev/null | tail -n 1 | sed 's/^[[:space:]]*//' | cut -c1-180); [ -n \"\$progress\" ] || progress=-; printf '%-30s %-10s %s\\n' \"\$id\" \"\$state\" \"\$progress\"; done"
    echo
}

show_host cs-cl-16 /localhome/mirzaei/fb/GraphVAE-REQ
show_host cs-cl-19 /local-scratch2/mirzaei/fb/GraphVAE-REQ
