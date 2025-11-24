#!/bin/bash
# ==========================================
#  Docker Helper Script for Flower Server
# ==========================================

set -euo pipefail

# Colors
GREEN="\e[32m"
YELLOW="\e[33m"
RED="\e[31m"
BLUE="\e[34m"
RESET="\e[0m"

# Default values
ACTION="${1:-help}"
PORT="${FL_PORT:-8080}"

show_help() {
    echo -e "${BLUE}============================================${RESET}"
    echo -e "${GREEN} Flower Docker Helper${RESET}"
    echo -e "${BLUE}============================================${RESET}\n"
    echo "Usage: ./docker-run.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker images (server + workers)"
    echo "  start       Start all containers (server + 2 workers)"
    echo "  stop        Stop all containers"
    echo "  restart     Restart all containers"
    echo "  logs        Show container logs (all or specific service)"
    echo "  status      Show container status"
    echo "  shell       Open a shell in a container (server/worker1/worker2)"
    echo "  clean       Stop and remove all containers/images"
    echo "  help        Show this help message"
    echo ""
    echo "Service-specific commands:"
    echo "  start-server    Start only the server"
    echo "  start-workers   Start only the workers"
    echo "  stop-server     Stop only the server"
    echo "  stop-workers    Stop only the workers"
    echo ""
    echo "Environment Variables:"
    echo "  FL_PORT              Server port (default: 8080)"
    echo "  FL_MIN_CLIENTS      Minimum clients (default: 2)"
    echo "  FL_NUM_ROUNDS       Number of rounds (default: 5)"
    echo ""
    echo "Examples:"
    echo "  ./docker-run.sh build"
    echo "  ./docker-run.sh start"
    echo "  ./docker-run.sh logs flower-server"
    echo "  ./docker-run.sh shell flower-worker-1"
    echo "  FL_PORT=9090 ./docker-run.sh start"
    echo ""
}

build_image() {
    echo -e "${BLUE} Building Docker images (server + workers)...${RESET}"
    docker-compose build
    echo -e "${GREEN}✓ Images built successfully${RESET}\n"
}

start_server() {
    echo -e "${BLUE} Starting all containers (server + 2 workers)...${RESET}"
    docker-compose up -d
    echo -e "${GREEN}✓ All containers started${RESET}"
    echo -e "  Server: ${YELLOW}flower-server${RESET} (port ${PORT})"
    echo -e "  Worker 1: ${YELLOW}flower-worker-1${RESET}"
    echo -e "  Worker 2: ${YELLOW}flower-worker-2${RESET}"
    echo -e "  Logs: ${YELLOW}docker-compose logs -f${RESET}\n"
}

stop_server() {
    echo -e "${BLUE} Stopping all containers...${RESET}"
    docker-compose down
    echo -e "${GREEN}✓ All containers stopped${RESET}\n"
}

restart_server() {
    echo -e "${BLUE} Restarting Flower server container...${RESET}"
    docker-compose restart
    echo -e "${GREEN}✓ Server restarted${RESET}\n"
}

show_logs() {
    SERVICE="${2:-}"
    if [ -n "$SERVICE" ]; then
        echo -e "${BLUE} Showing logs for ${SERVICE}...${RESET}"
        docker-compose logs -f "$SERVICE"
    else
        echo -e "${BLUE} Showing logs for all containers...${RESET}"
        docker-compose logs -f
    fi
}

show_status() {
    echo -e "${BLUE} Container status:${RESET}\n"
    docker-compose ps
    echo ""
    
    if docker ps | grep -q flower-server; then
        echo -e "${GREEN}✓ Server is running${RESET}"
        docker inspect flower-server --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' | xargs -I {} echo -e "  IP: ${YELLOW}{}${RESET}"
    else
        echo -e "${RED}✗ Server is not running${RESET}"
    fi
    
    WORKER_COUNT=$(docker ps | grep -c flower-worker || echo "0")
    if [ "$WORKER_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ ${WORKER_COUNT} worker(s) running${RESET}"
    else
        echo -e "${YELLOW}⚠ No workers running${RESET}"
    fi
    echo ""
}

open_shell() {
    SERVICE="${2:-flower-server}"
    echo -e "${BLUE} Opening shell in ${SERVICE}...${RESET}"
    docker-compose exec "$SERVICE" /bin/bash || docker exec -it "$SERVICE" /bin/bash
}

clean_all() {
    echo -e "${YELLOW} This will stop and remove all containers and images.${RESET}"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE} Cleaning up...${RESET}"
        docker-compose down -v --rmi all
        echo -e "${GREEN}✓ Cleanup complete${RESET}\n"
    else
        echo -e "${YELLOW} Cleanup cancelled${RESET}\n"
    fi
}

start_server_only() {
    echo -e "${BLUE} Starting only the server...${RESET}"
    docker-compose up -d flower-server
    echo -e "${GREEN}✓ Server started${RESET}\n"
}

start_workers_only() {
    echo -e "${BLUE} Starting only the workers...${RESET}"
    docker-compose up -d flower-worker-1 flower-worker-2
    echo -e "${GREEN}✓ Workers started${RESET}\n"
}

stop_server_only() {
    echo -e "${BLUE} Stopping only the server...${RESET}"
    docker-compose stop flower-server
    echo -e "${GREEN}✓ Server stopped${RESET}\n"
}

stop_workers_only() {
    echo -e "${BLUE} Stopping only the workers...${RESET}"
    docker-compose stop flower-worker-1 flower-worker-2
    echo -e "${GREEN}✓ Workers stopped${RESET}\n"
}

# Main command handler
case "$ACTION" in
    build)
        build_image
        ;;
    start)
        start_server
        ;;
    start-server)
        start_server_only
        ;;
    start-workers)
        start_workers_only
        ;;
    stop)
        stop_server
        ;;
    stop-server)
        stop_server_only
        ;;
    stop-workers)
        stop_workers_only
        ;;
    restart)
        restart_server
        ;;
    logs)
        show_logs "$@"
        ;;
    status)
        show_status
        ;;
    shell)
        open_shell "$@"
        ;;
    clean)
        clean_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED} Unknown command: $ACTION${RESET}\n"
        show_help
        exit 1
        ;;
esac

