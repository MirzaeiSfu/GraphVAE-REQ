#!/bin/bash
# usage: launch.sh DATASET TRAIN_SEED GEN_SEED GPU N [PY]
DS=$1; S=$2; G=$3; GPU=$4; N=$5
W=${SYNFIX_W:-/local-scratch2/mirzaei/synfix_defog_work}; B=$W/bundle
PY=${6:-/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python}
dl=$(echo $DS | tr A-Z a-z)
NAME=${DS}_seed${S}_g${G}
mkdir -p $W/runs
tmux new-session -d -s synfix_defog_${NAME} "cd $W && CUDA_VISIBLE_DEVICES=$GPU $PY $B/run_defog_sampling.py --defog-root $B/defog_source --graph-eval-src $B/graph_evaluation/src --data-root $B/data/$dl --dataset $DS --train-seed $S --ckpt $B/ckpt/${DS}_seed_${S}.ckpt --gen-seed $G --n $N --out-dir $W/runs/$NAME > $W/runs/$NAME.log 2>&1"
echo launched $NAME on $(hostname) GPU $GPU
