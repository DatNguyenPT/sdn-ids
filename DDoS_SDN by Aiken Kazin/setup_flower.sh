#!/bin/bash
# ==========================================
#  Flower Setup Script
# ==========================================

set -euo pipefail

# Colors
GREEN="\e[32m"
YELLOW="\e[33m"
RED="\e[31m"
BLUE="\e[34m"
RESET="\e[0m"

echo -e "${BLUE}============================================${RESET}"
echo -e "${GREEN} Flower (flwr) Setup${RESET}"
echo -e "${BLUE}============================================${RESET}\n"

# Check if virtual environment exists
VENV_PATH=".venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${BLUE} Activating virtual environment...${RESET}"
    if [ -f "$VENV_PATH/Scripts/activate" ]; then
        source "$VENV_PATH/Scripts/activate"
    elif [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
    fi
else
    echo -e "${YELLOW} Warning: Virtual environment not found at $VENV_PATH${RESET}"
    echo -e "${YELLOW} You may want to create one: python -m venv .venv${RESET}\n"
fi

# Check if Flower is already installed
if python -c "import flwr" 2>/dev/null; then
    FLOWER_VERSION=$(python -c "import flwr; print(flwr.__version__)" 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓ Flower is already installed (version: $FLOWER_VERSION)${RESET}\n"
else
    echo -e "${BLUE} Installing Flower...${RESET}"
    
    # Install from requirements.txt if it exists
    if [ -f "../requirement.txt" ]; then
        echo -e "${BLUE} Installing from requirement.txt...${RESET}"
        pip install -r ../requirement.txt
    else
        echo -e "${BLUE} Installing Flower directly...${RESET}"
        pip install flwr
    fi
    
    if python -c "import flwr" 2>/dev/null; then
        FLOWER_VERSION=$(python -c "import flwr; print(flwr.__version__)" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}✓ Flower installed successfully (version: $FLOWER_VERSION)${RESET}\n"
    else
        echo -e "${RED}✗ Failed to install Flower${RESET}"
        exit 1
    fi
fi

# Make server script executable
if [ -f "flower_server.py" ]; then
    chmod +x flower_server.py
    echo -e "${GREEN}✓ Server script is executable${RESET}\n"
fi

# Summary
echo -e "${BLUE}============================================${RESET}"
echo -e "${GREEN} Setup Complete!${RESET}"
echo -e "${BLUE}============================================${RESET}\n"

echo -e "To start the Flower server, run:"
echo -e "  ${YELLOW}python flower_server.py${RESET}\n"

echo -e "Or with custom settings:"
echo -e "  ${YELLOW}python flower_server.py --address 0.0.0.0 --port 8080${RESET}\n"

echo -e "To send model parameters, run:"
echo -e "  ${YELLOW}python send_params.py [model_name.h5]${RESET}\n"

echo -e "Or use the full pipeline:"
echo -e "  ${YELLOW}./train_evaluate.sh${RESET}\n"

