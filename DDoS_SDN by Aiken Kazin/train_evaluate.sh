#!/bin/bash
# ==========================================
#  Automated ML Training & Evaluation Pipeline
# ==========================================

set -euo pipefail
IFS=$'\n\t'

# === Configuration ===
VENV_PATH=".venv"
TRAIN_SCRIPT="train.py"
EVAL_SCRIPT="evaluate.py"
LOG_DIR="logs"
MODEL_DIR="models"
REPORT_DIR="report"
IMG_DIR="img"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# === ANSI colors ===
GREEN="\e[32m"
YELLOW="\e[33m"
RED="\e[31m"
BLUE="\e[34m"
RESET="\e[0m"

# === Setup environment ===
mkdir -p "$LOG_DIR" "$MODEL_DIR" "$REPORT_DIR" "$IMG_DIR"

echo -e "${BLUE}============================================${RESET}"
echo -e "${GREEN} Starting ML pipeline at $(date)${RESET}"
echo -e "${BLUE}============================================${RESET}"
echo -e "Logs → ${YELLOW}${LOG_FILE}${RESET}\n"

# === Activate virtual environment (cross-platform) ===
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -d "$VENV_PATH" ]; then
        echo -e "${BLUE} Activating virtual environment...${RESET}"
        if [ -f "$VENV_PATH/Scripts/activate" ]; then
            # Windows-style venv
            source "$VENV_PATH/Scripts/activate"
        elif [ -f "$VENV_PATH/bin/activate" ]; then
            # Linux/macOS-style venv
            source "$VENV_PATH/bin/activate"
        else
            echo -e "${RED} Could not find activate script in $VENV_PATH${RESET}"
            exit 1
        fi
    else
        echo -e "${RED} Virtual environment not found at $VENV_PATH${RESET}"
        exit 1
    fi
else
    echo -e "${GREEN} Already inside virtual environment${RESET}"
fi

# === Run training ===
echo -e "\n${BLUE} Running training script...${RESET}"
python "$TRAIN_SCRIPT" | tee -a "$LOG_FILE"

# === Detect newest model file ===
LATEST_MODEL=$(ls -t "${MODEL_DIR}"/*.h5 2>/dev/null | head -n 1 || true)
if [ -z "$LATEST_MODEL" ]; then
    echo -e "${RED} No model (.h5) found in ${MODEL_DIR}${RESET}"
    exit 1
else
    MODEL_NAME=$(basename "$LATEST_MODEL")
    echo -e "${GREEN} Latest model detected: ${MODEL_NAME}${RESET}"
fi

# === Run evaluation ===
echo -e "\n${BLUE} Running evaluation script for ${MODEL_NAME}...${RESET}"
python "$EVAL_SCRIPT" "$MODEL_NAME" | tee -a "$LOG_FILE"

# === Verify outputs ===
echo -e "\n${BLUE} Verifying outputs...${RESET}"

if ls "${REPORT_DIR}"/evaluation_report_*.xlsx >/dev/null 2>&1; then
    echo -e "${GREEN} Evaluation report generated${RESET}"
else
    echo -e "${YELLOW}  No evaluation report found${RESET}"
fi

if ls "${IMG_DIR}"/confusion_matrix_*.png >/dev/null 2>&1; then
    echo -e "${GREEN} Confusion matrix image generated${RESET}"
else
    echo -e "${YELLOW}  No confusion matrix image found${RESET}"
fi

# === Completion summary ===
echo -e "\n${BLUE}============================================${RESET}"
echo -e "${GREEN} Pipeline completed successfully!${RESET}"
echo -e " Logs:    ${YELLOW}${LOG_FILE}${RESET}"
echo -e " Models:  ${YELLOW}${MODEL_DIR}${RESET}"
echo -e " Reports: ${YELLOW}${REPORT_DIR}${RESET}"
echo -e " Images:  ${YELLOW}${IMG_DIR}${RESET}"
echo -e "${BLUE}============================================${RESET}\n"
