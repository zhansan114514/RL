#!/usr/bin/env bash
# ACC-Collab: Auto-detect free GPU, run full pipeline, monitor every 10 min
# Features:
#   - Finds free H100, runs 06_full_pipeline.py (one-shot)
#   - All outputs under cache/ (models, trajectories, preferences, content, logs)
#   - Monitors every 10 min, auto-fixes and restarts on failure
#   - Lock file prevents duplicate runs
#
# Usage: bash scripts/monitor_and_run.sh
# Monitor log: tail -f cache/logs/monitor.log

set -euo pipefail

PROJECT_ROOT="/home/storage/zyw/RL"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export LD_LIBRARY_PATH="/home/shichen/miniconda3/lib:${LD_LIBRARY_PATH:-}"

# Read cache_dir from config (default: cache)
CONFIG="configs/experiment_qwen3_arc.yaml"
CACHE_DIR=$(python -c "
import yaml, os
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg.get('common', {}).get('cache_dir', 'cache'))
" 2>/dev/null || echo "cache")

LOG_DIR="$CACHE_DIR/logs"
mkdir -p "$LOG_DIR" "$CACHE_DIR/trajectories" "$CACHE_DIR/models" "$CACHE_DIR/content"

LOCK_FILE="$CACHE_DIR/logs/pipeline.lock"
MONITOR_LOG="$LOG_DIR/monitor.log"
PIPELINE_LOG="$LOG_DIR/full_pipeline.log"
MAX_RESTARTS=5
MIN_FREE_GB=45

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"; }

# ── Kill any stale experiment processes ──
kill_stale() {
    if [ -f "$LOCK_FILE" ]; then
        OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
        if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
            log "Killing stale process PID=$OLD_PID"
            kill "$OLD_PID" 2>/dev/null || true
            sleep 3
            kill -9 "$OLD_PID" 2>/dev/null || true
        fi
        rm -f "$LOCK_FILE"
    fi
}

# ── Check if results already exist (pipeline completed successfully) ──
check_completed() {
    if [ -f "$CACHE_DIR/full_pipeline/results.json" ] || [ -f "$CACHE_DIR/results.json" ]; then
        log "Pipeline already completed (results.json found). Exiting."
        exit 0
    fi
}

# ── Find the GPU with most free memory ──
find_gpu() {
    BEST_GPU=""
    BEST_FREE=0
    while IFS=, read -r idx mem_used mem_free mem_total util; do
        free_gb=$(echo "$mem_free" | awk '{printf "%.0f", $1/1024}')
        if [ "$free_gb" -gt "$BEST_FREE" ]; then
            BEST_FREE=$free_gb
            BEST_GPU=$(echo "$idx" | tr -d ' ')
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader)
    echo "$BEST_GPU $BEST_FREE"
}

# ── Kill residual vLLM EngineCore processes on a specific GPU ──
kill_gpu_residuals() {
    local gpu_id=$1
    for pid in $(fuser -v /dev/nvidia${gpu_id} 2>&1 | grep -oP '\d+' || true); do
        # Only kill vLLM EngineCore processes, not other users' work
        if ps -p "$pid" -o comm= 2>/dev/null | grep -qi "VLLM\|EngineCor"; then
            log "Killing residual vLLM process PID=$pid on GPU $gpu_id"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}

# ── Launch the full pipeline on given GPU ──
launch_pipeline() {
    local gpu_id=$1
    export CUDA_VISIBLE_DEVICES="$gpu_id"

    log "Launching full pipeline on GPU $gpu_id"
    log "Config: $CONFIG, Cache: $CACHE_DIR"
    log "Log file: $PIPELINE_LOG"

    nohup python scripts/06_full_pipeline.py \
        --config "$CONFIG" \
        >> "$PIPELINE_LOG" 2>&1 &

    local pid=$!
    echo "$pid" > "$LOCK_FILE"
    log "Pipeline started PID=$pid on GPU=$gpu_id"
}

# ── Check process health ──
check_health() {
    if [ ! -f "$LOCK_FILE" ]; then
        echo "NO_LOCK"
        return
    fi
    local pid
    pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -z "$pid" ]; then
        echo "EMPTY_LOCK"
        return
    fi
    if kill -0 "$pid" 2>/dev/null; then
        echo "RUNNING $pid"
    else
        echo "DEAD $pid"
    fi
}

# ── Diagnose failure from log ──
diagnose_and_fix() {
    log "Diagnosing failure..."
    local tail_log
    tail_log=$(tail -50 "$PIPELINE_LOG" 2>/dev/null || echo "no log")

    if echo "$tail_log" | grep -qi "out of memory\|CUDA out of memory"; then
        log "Detected OOM. Killing GPU residuals..."
        kill_gpu_residuals "$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1 | tr -d ' ')"
        return 0
    fi

    if echo "$tail_log" | grep -qi "vllm\|RuntimeError\|ConnectionError"; then
        log "Detected vLLM/Runtime error. Clearing caches..."
        kill_gpu_residuals "$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1 | tr -d ' ')"
        return 0
    fi

    if echo "$tail_log" | grep -qi "ImportError\|ModuleNotFoundError"; then
        log "Detected import error. Verifying PYTHONPATH=$PYTHONPATH"
        return 0
    fi

    log "Unknown failure. Will retry."
    return 0
}

# ────────────────── Main Loop ──────────────────

restart_count=0
log "=========================================="
log " ACC-Collab Monitor Started"
log " Config: $CONFIG"
log " Cache dir: $CACHE_DIR"
log " Max restarts: $MAX_RESTARTS"
log " Min GPU free: ${MIN_FREE_GB}GB"
log "=========================================="

check_completed

# Initial launch
read -r gpu free_gb <<< "$(find_gpu)"
log "Best GPU: $gpu (${free_gb}GB free)"

if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    log "No GPU has enough memory (need ${MIN_FREE_GB}GB). Waiting..."
else
    kill_stale
    launch_pipeline "$gpu"
    sleep 60
    read -r status _ <<< "$(check_health)"
    if [ "$status" != "RUNNING" ]; then
        log "Process died within 60s!"
        tail -20 "$PIPELINE_LOG" | tee -a "$MONITOR_LOG"
        restart_count=$((restart_count + 1))
    fi
fi

# ── Monitor loop ──
while true; do
    sleep 300  # 5 minutes

    check_completed

    read -r status pid <<< "$(check_health)"

    case "$status" in
        RUNNING)
            log "Heartbeat: PID=$pid alive. Trajectories:"
            ls -la "$CACHE_DIR/trajectories/" 2>/dev/null | tail -5 | tee -a "$MONITOR_LOG"
            ;;
        NO_LOCK|EMPTY_LOCK)
            if [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
                log "Max restarts ($MAX_RESTARTS) reached. Giving up."
                exit 1
            fi
            read -r gpu free_gb <<< "$(find_gpu)"
            if [ "$free_gb" -ge "$MIN_FREE_GB" ]; then
                log "No process running. Restarting on GPU $gpu ($((restart_count+1))/$MAX_RESTARTS)"
                launch_pipeline "$gpu"
                restart_count=$((restart_count + 1))
            else
                log "No process running but no free GPU. Waiting..."
            fi
            ;;
        DEAD)
            if [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
                log "Max restarts ($MAX_RESTARTS) reached. Giving up."
                log "Last 30 lines of pipeline log:"
                tail -30 "$PIPELINE_LOG" | tee -a "$MONITOR_LOG"
                exit 1
            fi
            log "Process $pid died! Diagnosing..."
            tail -20 "$PIPELINE_LOG" | tee -a "$MONITOR_LOG"
            if diagnose_and_fix; then
                rm -f "$LOCK_FILE"
                read -r gpu free_gb <<< "$(find_gpu)"
                if [ "$free_gb" -ge "$MIN_FREE_GB" ]; then
                    log "Restarting on GPU $gpu ($((restart_count+1))/$MAX_RESTARTS)"
                    launch_pipeline "$gpu"
                    restart_count=$((restart_count + 1))
                else
                    log "No free GPU for restart. Will retry next cycle."
                fi
            else
                log "Unrecoverable error. Stopping."
                exit 1
            fi
            ;;
    esac
done
