#!/bin/bash
# =============================================================================
# AgentBoard Evaluation Script
# Evaluates all 8 combinations from the exp.md table on AlfWorld, ScienceWorld,
# and BabyAI using the zzh202121/agentboard:0117 Docker image.
#
# Table rows:
#   Execute     | Plan         | Config
#   ------------|--------------|----------------------------------------------
#   Qwen-8B     | w/o          | onepass.yaml            --model qwen3
#   Qwen-8B     | Qwen-8B      | collaboration_qwen3_qwen3.yaml
#   Qwen-8B     | Llada-8B     | collaboration_llm_dllm.yaml (llada)
#   Qwen-8B     | Dream-7B     | collaboration_qwen3_dream.yaml
#   Ministral-8B| w/o          | onepass.yaml            --model ministral
#   Ministral-8B| Ministral-8B | collaboration_ministral_ministral.yaml
#   Ministral-8B| Llada-8B     | collaboration_ministral_llada.yaml
#   Ministral-8B| Dream-7B     | collaboration_ministral_dream.yaml
#
# LLM serving:  SGLang (bfcl conda env) on port 23456
# DLLM serving: custom FastAPI server (dllm conda env) on port 23450
# Eval runtime: Docker container zzh202121/agentboard:0117 with --network host
#
# Usage:
#   bash run_agentboard_eval.sh [--rows "1 2 3"] [--tasks "alfworld scienceworld babyai"]
#   bash run_agentboard_eval.sh            # run all 8 rows, all 3 tasks
#   bash run_agentboard_eval.sh --rows "1 5"   # run row 1 (qwen3 w/o) and row 5 (ministral w/o)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PATH="${REPO_ROOT}/unified_envs/AgentBoard"
AGENTBOARD_DIR="${PROJECT_PATH}/agentboard"
OUTPUTS_DIR="${PROJECT_PATH}/outputs"

CONDA_BASE="/home/u7444045/miniconda3"
BFCL_PYTHON="${CONDA_BASE}/envs/bfcl/bin/python"
DLLM_PYTHON="${CONDA_BASE}/envs/dllm/bin/python"

DOCKER_IMAGE="zzh202121/agentboard:0117"

QWEN3_PATH="/data/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
MINISTRAL_PATH="/data/.cache/huggingface/hub/models--mistralai--Ministral-8B-Instruct-2410/snapshots/2f494a194c5b980dfb9772cb92d26cbb671fce5a"

LLM_PORT=23456
DLLM_PORT=23450
LLM_BASE_URL="http://localhost:${LLM_PORT}/"
DLLM_BASE_URL="http://localhost:${DLLM_PORT}/"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
ROWS_TO_RUN="${1:-all}"
TASKS_TO_RUN="alfworld_enhanced scienceworld_enhanced babyai_enhanced"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rows) ROWS_TO_RUN="$2"; shift 2 ;;
        --tasks) TASKS_TO_RUN="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ "$ROWS_TO_RUN" == "all" ]]; then
    ROWS_TO_RUN="1 2 3 4 5 6 7 8"
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
    local url="$1"
    local name="$2"
    local pid="${3:-}"
    local log_file="${4:-}"
    local max_wait=400
    log "Waiting for ${name} at ${url} ..."
    for i in $(seq 1 $max_wait); do
        if curl -sf "${url}health" -o /dev/null 2>/dev/null || \
           curl -sf "${url}v1/models" -o /dev/null 2>/dev/null; then
            log "${name} is ready."
            return 0
        fi
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            log "ERROR: ${name} exited before becoming ready."
            if [[ -n "$log_file" && -f "$log_file" ]]; then
                log "Last lines from ${log_file}:"
                tail -n 40 "$log_file" || true
            fi
            return 1
        fi
        sleep 2
    done
    log "ERROR: ${name} did not start within ${max_wait}s"
    if [[ -n "$log_file" && -f "$log_file" ]]; then
        log "Last lines from ${log_file}:"
        tail -n 40 "$log_file" || true
    fi
    return 1
}

kill_port() {
    local port="$1"
    local pid
    pid=$(lsof -ti ":${port}" 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
        # Wait up to 30s for GPU memory to be released before letting the next
        # server start; plain sleep 2 was too short and caused CUDA OOM.
        local waited=0
        while kill -0 "$pid" 2>/dev/null && (( waited < 30 )); do
            sleep 2; (( waited += 2 ))
        done
        sleep 5  # extra buffer for CUDA context teardown
        log "Killed process on port ${port} (waited ${waited}s for exit)"
    fi
}

start_llm_server() {
    local model_path="$1"
    local model_name="$2"
    local log_file="${OUTPUTS_DIR}/llm_server_${model_name}.log"
    kill_port $LLM_PORT
    log "Starting SGLang LLM server for ${model_name} on port ${LLM_PORT} ..."

    # Qwen3 uses hybrid thinking mode; --reasoning-parser qwen3 routes think tokens
    # to a separate field so that response 'content' contains only the clean answer.
    local extra_flags=""
    if [[ "$model_name" == "qwen3" ]]; then
        extra_flags="--reasoning-parser qwen3"
    fi

    "${BFCL_PYTHON}" -m sglang.launch_server \
        --model-path "${model_path}" \
        --served-model-name "${model_path}" \
        --port ${LLM_PORT} \
        --host 0.0.0.0 \
        --trust-remote-code \
        --mem-fraction-static 0.50 \
        --quantization fp8 \
        ${extra_flags} \
        > "${log_file}" 2>&1 &
    LLM_SERVER_PID=$!
    log "LLM server PID: ${LLM_SERVER_PID}"
    wait_for_server "${LLM_BASE_URL}" "LLM server (${model_name})" "${LLM_SERVER_PID}" "${log_file}"
}

start_dllm_server() {
    local dllm_type="$1"  # "dream" or "llada"
    local log_file="${OUTPUTS_DIR}/dllm_server_${dllm_type}.log"
    kill_port $DLLM_PORT
    log "Starting DiffusionLLM server for ${dllm_type} on port ${DLLM_PORT} ..."
    "${DLLM_PYTHON}" "${REPO_ROOT}/dllm_server.py" \
        --model "${dllm_type}" \
        --port ${DLLM_PORT} \
        > "${log_file}" 2>&1 &
    DLLM_SERVER_PID=$!
    log "DLLM server PID: ${DLLM_SERVER_PID}"
    wait_for_server "${DLLM_BASE_URL}" "DLLM server (${dllm_type})" "${DLLM_SERVER_PID}" "${log_file}"
}

stop_server() {
    local pid="${1:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        sleep 2
    fi
}

# ---------------------------------------------------------------------------
# Data download check
# ---------------------------------------------------------------------------
check_data() {
    if [[ ! -d "${AGENTBOARD_DIR}/data" ]]; then
        log "AgentBoard data directory not found. Downloading from HuggingFace..."
        mkdir -p "${OUTPUTS_DIR}"
        cd "${AGENTBOARD_DIR}"
        wget -q --show-progress \
            "https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz" \
            -O /tmp/agentboard_data.tar.gz
        tar -xzvf /tmp/agentboard_data.tar.gz -C "${AGENTBOARD_DIR}/"
        rm -f /tmp/agentboard_data.tar.gz
        log "Data downloaded and extracted."
        cd "${REPO_ROOT}"
    else
        log "AgentBoard data directory found."
    fi
}

# ---------------------------------------------------------------------------
# Docker eval runner
# ---------------------------------------------------------------------------
run_eval() {
    local config="$1"
    local model="$2"
    local log_suffix="$3"
    local log_path="/project/outputs/${log_suffix}"

    log "Running eval: config=${config} model=${model} log=${log_suffix}"

    # Mount project at its real host path so absolute paths in base_config.yaml work
    docker run --rm \
        --network host \
        -v "${PROJECT_PATH}:${PROJECT_PATH}" \
        -v "/data:/data" \
        -e PROJECT_PATH="${PROJECT_PATH}" \
        -e MAIN_AGENT_BASE_URL="${LLM_BASE_URL}" \
        -e MAIN_AGENT_API_KEY="none" \
        -e DLLM_BASE_URL="${DLLM_BASE_URL}" \
        -e DLLM_API_KEY="none" \
        "${DOCKER_IMAGE}" \
        bash -c "
            source /root/miniconda3/etc/profile.d/conda.sh && \
            conda activate agentboard && \
            cd ${PROJECT_PATH}/agentboard && \
            python eval_modular.py \
                --cfg-path ${PROJECT_PATH}/agentboard/configs/experiments/${config} \
                --model ${model} \
                --tasks ${TASKS_TO_RUN} \
                --max_num_steps 30 \
                --log_path ${OUTPUTS_DIR}/${log_suffix}
        " 2>&1 | tee "${OUTPUTS_DIR}/${log_suffix}.log"

    log "Eval finished: ${log_suffix}"
}

# ---------------------------------------------------------------------------
# Experiment definitions
# Format: "ROW_NUM|LLM_MODEL|LLM_PATH|DLLM_TYPE|CONFIG|MODEL_FLAG|LOG_SUFFIX"
# DLLM_TYPE: "none", "dream", or "llada"
# ---------------------------------------------------------------------------
declare -A EXPERIMENTS
EXPERIMENTS[1]="qwen3|${QWEN3_PATH}|none|onepass.yaml|qwen3|qwen3_woplanning"
EXPERIMENTS[2]="qwen3|${QWEN3_PATH}|none|collaboration_qwen3_qwen3.yaml|qwen3|qwen3_plan_qwen3"
EXPERIMENTS[3]="qwen3|${QWEN3_PATH}|llada|collaboration_llm_dllm.yaml|qwen3|qwen3_plan_llada"
EXPERIMENTS[4]="qwen3|${QWEN3_PATH}|dream|collaboration_qwen3_dream.yaml|qwen3|qwen3_plan_dream"
EXPERIMENTS[5]="ministral|${MINISTRAL_PATH}|none|onepass.yaml|ministral|ministral_woplanning"
EXPERIMENTS[6]="ministral|${MINISTRAL_PATH}|none|collaboration_ministral_ministral.yaml|ministral|ministral_plan_ministral"
EXPERIMENTS[7]="ministral|${MINISTRAL_PATH}|llada|collaboration_ministral_llada.yaml|ministral|ministral_plan_llada"
EXPERIMENTS[8]="ministral|${MINISTRAL_PATH}|dream|collaboration_ministral_dream.yaml|ministral|ministral_plan_dream"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUTS_DIR}"
check_data

LLM_SERVER_PID=""
DLLM_SERVER_PID=""
CURRENT_LLM=""
CURRENT_DLLM=""

trap 'log "Caught exit signal, stopping servers..."; stop_server "$LLM_SERVER_PID"; stop_server "$DLLM_SERVER_PID"; kill_port $LLM_PORT; kill_port $DLLM_PORT' EXIT

for row in $ROWS_TO_RUN; do
    if [[ -z "${EXPERIMENTS[$row]+x}" ]]; then
        log "WARNING: Row ${row} not defined, skipping."
        continue
    fi

    IFS='|' read -r llm_name llm_path dllm_type config model_flag log_suffix \
        <<< "${EXPERIMENTS[$row]}"

    log "============================================================"
    log "Row ${row}: Execute=${llm_name}, Plan=${dllm_type:-none}"
    log "  Config: ${config}"
    log "  Model:  ${model_flag}"
    log "============================================================"

    # --- Start / restart LLM server if model changed ---
    if [[ "$CURRENT_LLM" != "$llm_name" ]]; then
        stop_server "$LLM_SERVER_PID"; LLM_SERVER_PID=""
        kill_port $LLM_PORT
        start_llm_server "${llm_path}" "${llm_name}"
        CURRENT_LLM="$llm_name"
    fi

    # --- Start / restart DLLM server if type changed ---
    if [[ "$dllm_type" == "none" ]]; then
        stop_server "$DLLM_SERVER_PID"; DLLM_SERVER_PID=""
        kill_port $DLLM_PORT
        CURRENT_DLLM=""
    elif [[ "$CURRENT_DLLM" != "$dllm_type" ]]; then
        stop_server "$DLLM_SERVER_PID"; DLLM_SERVER_PID=""
        kill_port $DLLM_PORT
        start_dllm_server "$dllm_type"
        CURRENT_DLLM="$dllm_type"
    fi

    # --- Run eval in Docker ---
    run_eval "${config}" "${model_flag}" "${log_suffix}"

done

log "============================================================"
log "All rows done. Results in ${OUTPUTS_DIR}/"
log "============================================================"

# Print summary (success/progress rates from log files)
echo ""
echo "=== Results Summary ==="
for row in $ROWS_TO_RUN; do
    [[ -z "${EXPERIMENTS[$row]+x}" ]] && continue
    IFS='|' read -r llm_name _ _ _ _ log_suffix <<< "${EXPERIMENTS[$row]}"
    result_file="${OUTPUTS_DIR}/${log_suffix}.log"
    if [[ -f "$result_file" ]]; then
        echo "--- Row ${row}: ${log_suffix} ---"
        grep -E "success_rate|progress_rate|Success Rate|Progress Rate" "$result_file" 2>/dev/null | tail -20 || echo "  (no summary found)"
    fi
done
