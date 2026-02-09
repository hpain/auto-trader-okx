#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}Targeted Restart: Updating Discord Bot & Supervisor...${NC}"

# --- Compatibility Fixes (from deploy/start.sh) ---
# Fix: "KeyError: 'ContainerConfig'" for legacy docker-compose (1.29.x)
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

# Determine docker compose command
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    DOCKER_COMPOSE_CMD="docker-compose"
fi

# --- Execution ---
echo -e "${YELLOW}Rebuilding and restarting non-trading services ONLY...${NC}"
echo -e "${YELLOW}Executing: $DOCKER_COMPOSE_CMD up -d --build --no-deps discord-bot llm-supervisor${NC}"

$DOCKER_COMPOSE_CMD up -d --build --no-deps discord-bot llm-supervisor

echo -e "${GREEN}Update Successful!${NC}"
echo -e "${GREEN}Live trading bots (ML and Arb) were NOT affected.${NC}"
