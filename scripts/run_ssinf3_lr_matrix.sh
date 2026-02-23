#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source ../SSINF/exodus2/.venv/bin/activate

if [[ $# -ge 1 ]]; then
  RUN_DIR="$1"
else
  STAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="logs/lr_matrix_${STAMP}"
fi
mkdir -p "$RUN_DIR"
SUMMARY_TSV="$RUN_DIR/summary.tsv"

BASE_LR="0.08"
LOW_LR="0.06"
HIGH_LR="0.10"

# 10 permutations (5 groups x {low,high}); anchors kept, scalars omitted due user cap.
LR_GROUPS=(global_mlp local_mlp basis tangents anchors)

if [[ ! -f "$SUMMARY_TSV" ]]; then
  cat > "$SUMMARY_TSV" <<EOF
run_name	group	mode	group_lr	train_loss	val_loss	step_avg_ms	peak_mem_mib	log
EOF
fi

cp train.py _tmp_train_100.py
perl -0pi -e 's/batch_size\s*:\s*int\s*=\s*8\*64/batch_size : int = 64/; s/device_batch_size\s*:\s*int\s*=\s*64/device_batch_size : int = 1/; s/num_iterations\s*:\s*int\s*=\s*3000/num_iterations : int = 100/; s/val_loss_every\s*:\s*int\s*=\s*125/val_loss_every : int = 0/; s/val_tokens\s*:\s*int\s*=\s*10485760/val_tokens : int = 1024/; s#model = torch\.compile\(model\)#\# model = torch.compile(model)#; s#/workspace/triton-cache#/tmp/triton-cache#g; s#/workspace/torchinductor-cache#/tmp/torchinductor-cache#g' _tmp_train_100.py

for group in "${LR_GROUPS[@]}"; do
  for mode in low high; do
    if [[ "$mode" == "low" ]]; then
      lr="$LOW_LR"
    else
      lr="$HIGH_LR"
    fi

    run_name="${group}_${mode}"
    log_file="${RUN_DIR}/${run_name}.out"

    if rg -q "^${run_name}[[:space:]]" "$SUMMARY_TSV"; then
      echo "==== Skipping ${run_name} (already completed) ===="
      continue
    fi

    export SSINF3_OPTIMIZED_BASE_LR="$BASE_LR"
    unset SSINF3_LR_GLOBAL_MLP SSINF3_LR_LOCAL_MLP SSINF3_LR_BASIS SSINF3_LR_ANCHORS SSINF3_LR_TANGENTS SSINF3_LR_SCALARS
    case "$group" in
      global_mlp) export SSINF3_LR_GLOBAL_MLP="$lr" ;;
      local_mlp)  export SSINF3_LR_LOCAL_MLP="$lr" ;;
      basis)      export SSINF3_LR_BASIS="$lr" ;;
      anchors)    export SSINF3_LR_ANCHORS="$lr" ;;
      tangents)   export SSINF3_LR_TANGENTS="$lr" ;;
      *)
        echo "unknown group: $group" >&2
        exit 1
        ;;
    esac

    echo "==== Running ${run_name} (base=${BASE_LR}, ${group}=${lr}) ===="
    torchrun --standalone --nproc_per_node=1 _tmp_train_100.py --ssinf3-weight-matched --ssinf3-optimized > "$log_file" 2>&1

    train_loss="$(rg 'step:100/100 train_loss:' "$log_file" | tail -n1 | sed -E 's/.*train_loss:([0-9.]+).*/\1/')"
    val_loss="$(rg 'step:100/100 val_loss:' "$log_file" | tail -n1 | sed -E 's/.*val_loss:([0-9.]+).*/\1/')"
    step_avg="$(rg 'step:100/100 val_loss:' "$log_file" | tail -n1 | sed -E 's/.*step_avg:([0-9.]+)ms.*/\1/')"
    peak_mem="$(rg 'peak memory consumption:' "$log_file" | tail -n1 | sed -E 's/.*: ([0-9]+) MiB/\1/')"

    echo -e "${run_name}\t${group}\t${mode}\t${lr}\t${train_loss}\t${val_loss}\t${step_avg}\t${peak_mem}\t${log_file}" >> "$SUMMARY_TSV"
    echo "==== Completed ${run_name}: train=${train_loss} val=${val_loss} step_avg_ms=${step_avg} peak_mem_mib=${peak_mem} ===="
  done
done

echo "Completed sweep. Summary: ${SUMMARY_TSV}"
