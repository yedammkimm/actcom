#!/usr/bin/env bash
# Phase 2 master chain (2026-08-21):
#   [1] 70B L=2048 oamp E2M1 memory re-measure (~30 min)
#       — resolves the sec/step outlier flagged in the initial sweep
#   [2] 70B accuracy: naive_fp4, oamp × seed 42 (~11 h)
#   [3] 3B accuracy:  naive_fp4, oamp × seed {123, 456} (~13 h)
#
# Chain does NOT abort on non-zero exits; each stage is logged and any failure
# is left to be triaged after chain end.

set -u
CONTAINER=hma-container

log() { printf '[%s] %s\n' "$(date +%m-%d\ %H:%M:%S)" "$*"; }

# --------------------------------------------------------------------------
# STAGE 1: 70B L=2048 oamp E2M1 memory re-measure
# --------------------------------------------------------------------------
STAGE1_OUT=results/70b_mem_sweep_e2m1
STAGE1_LOG=/home/yedam/HMA/HMA_Project/${STAGE1_OUT}/oamp_L2048_remeasure.log

log "===================================================="
log "STAGE 1: 70B L=2048 oamp E2M1 memory re-measure"
log "===================================================="
docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${STAGE1_OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${STAGE1_OUT} 2>/dev/null

docker exec ${CONTAINER} bash -c \
    "cd /app/HMA_Project && python run_experiment.py \
     --method oamp --fp8_ratio 0.2 --group_size 128 --mask_seed 42 \
     --body_encoding e2m1 \
     --mode memory --model 70B --weight_quant nf4 --bf16_rmsnorm \
     --pack_4d_mode fp8 \
     --mem_batch_size 1 --mem_steps 7 --mem_warmup 3 \
     --seed 42 --output_dir ${STAGE1_OUT} \
     --mem_seq_len 2048 --mem_abort_gb 100" \
    > "${STAGE1_LOG}" 2>&1
log "  stage1 exit=$?"
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${STAGE1_OUT} 2>/dev/null
sleep 60

# --------------------------------------------------------------------------
# STAGE 2: 70B accuracy naive_fp4 + oamp seed 42
# --------------------------------------------------------------------------
STAGE2_OUT=results/70b_accuracy_e2m1
STAGE2_LOG_HOST=/home/yedam/HMA/HMA_Project/${STAGE2_OUT}

log "===================================================="
log "STAGE 2: 70B accuracy naive_fp4 + oamp × seed 42"
log "===================================================="
mkdir -p ${STAGE2_LOG_HOST}
docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${STAGE2_OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${STAGE2_OUT} 2>/dev/null

COMMON_70B="--mode accuracy --model 70B --weight_quant nf4 --bf16_rmsnorm \
--body_encoding e2m1 --pack_4d_mode fp8 --seed 42 --task gsm8k \
--n_train 2000 --epochs 1 --batch_size 1 --grad_accum_steps 4 \
--max_seq_len 512 --lr 5e-5 --optimizer paged_adamw8bit \
--scheduler cosine --warmup_ratio 0.0 \
--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --grad_clip 1.0 \
--eval_samples 200 --eval_fewshot 8 --eval_max_new_tokens 256 \
--checkpoint_every 25 --mem_abort_gb 80 \
--output_dir ${STAGE2_OUT}"

for method in naive_fp4 oamp; do
    tag="${method}_seed42"
    log_file="${STAGE2_LOG_HOST}/${tag}.log"
    if [ "${method}" = "oamp" ]; then
        extras="--fp8_ratio 0.2 --group_size 128 --mask_seed 42"
    else
        extras=""
    fi
    log ""
    log "--- 70B accuracy: ${tag} ---"
    docker exec ${CONTAINER} bash -c \
        "cd /app/HMA_Project && python run_experiment.py \
         --method ${method} ${extras} ${COMMON_70B}" \
        > "${log_file}" 2>&1
    log "  ${tag} exit=$?"
    docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${STAGE2_OUT} 2>/dev/null
    sleep 60
done

# --------------------------------------------------------------------------
# STAGE 3: 3B accuracy naive_fp4 + oamp × seed {123, 456}
# --------------------------------------------------------------------------
STAGE3_OUT=results/pilot_e2m1_seeds
STAGE3_LOG_HOST=/home/yedam/HMA/HMA_Project/${STAGE3_OUT}

log "===================================================="
log "STAGE 3: 3B accuracy naive_fp4 + oamp × seed {123, 456}"
log "===================================================="
mkdir -p ${STAGE3_LOG_HOST}
docker exec ${CONTAINER} mkdir -p /app/HMA_Project/${STAGE3_OUT}
docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${STAGE3_OUT} 2>/dev/null

COMMON_3B="--mode accuracy --model 3B --weight_quant nf4 --bf16_rmsnorm \
--body_encoding e2m1 --pack_4d_mode fp8 --task gsm8k \
--n_train 7473 --epochs 2 --lr 2e-4 \
--batch_size 1 --grad_accum_steps 4 --max_seq_len 512 \
--optimizer paged_adamw8bit --scheduler cosine --warmup_ratio 0.0 \
--lora_dropout 0.05 --eval_samples 500 \
--output_dir ${STAGE3_OUT}"

for seed in 123 456; do
    for method in naive_fp4 oamp; do
        tag="${method}_seed${seed}"
        log_file="${STAGE3_LOG_HOST}/${tag}.log"
        if [ "${method}" = "oamp" ]; then
            extras="--fp8_ratio 0.2 --group_size 128 --mask_seed ${seed}"
        else
            extras=""
        fi
        log ""
        log "--- 3B accuracy: ${tag} ---"
        docker exec ${CONTAINER} bash -c \
            "cd /app/HMA_Project && python run_experiment.py \
             --method ${method} ${extras} --seed ${seed} ${COMMON_3B}" \
            > "${log_file}" 2>&1
        log "  ${tag} exit=$?"
        docker exec ${CONTAINER} chown -R 4051:4051 /app/HMA_Project/${STAGE3_OUT} 2>/dev/null
        sleep 30
    done
done

log ""
log "===================================================="
log "Phase 2 master chain complete"
log "===================================================="
