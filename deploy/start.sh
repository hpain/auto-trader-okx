#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}Starting Deployment for Auto Trader...${NC}"

# Check for .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}.env file not found. Creating from .env.example...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${YELLOW}Please edit .env with your API credentials.${NC}"
        # Optional: Interactive setup
        read -p "Do you want to enter your OKX API Key now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            read -p "API Key: " api_key
            read -p "Secret Key: " secret_key
            read -p "Passphrase: " passphrase
            
            # Simple sed replacement (works on Linux/Mac)
            sed -i "s/OKX_API_KEY=your_api_key_here/OKX_API_KEY=$api_key/" .env
            sed -i "s/OKX_SECRET_KEY=your_secret_key_here/OKX_SECRET_KEY=$secret_key/" .env
            sed -i "s/OKX_PASSPHRASE=your_passphrase_here/OKX_PASSPHRASE=$passphrase/" .env
        fi
    else
        echo -e "${RED}Error: .env.example not found!${NC}"
        exit 1
    fi
fi

# Check for config/settings.yaml
if [ ! -f config/settings.yaml ]; then
    echo -e "${YELLOW}config/settings.yaml not found. Creating from example...${NC}"
    if [ -f config/settings.yaml.example ]; then
        cp config/settings.yaml.example config/settings.yaml
    else
         echo -e "${RED}Error: config/settings.yaml.example not found!${NC}"
         exit 1
    fi
fi

# Ensure data directories exist (to avoid permission issues if docker creates them as root)
echo "Ensuring directory structure..."
mkdir -p logs models data/history data/cache charts

# Build and Start
# Build and Start
echo -e "${GREEN}Building and Starting Docker Containers...${NC}"

if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
    echo "Using: Docker Compose V2 ($DOCKER_COMPOSE_CMD)"
else
    # Fallback to legacy docker-compose
    DOCKER_COMPOSE_CMD="docker-compose"
    echo -e "${YELLOW}Warning: Docker Compose V2 not found. Using legacy docker-compose.${NC}"
    
    # FIX: "KeyError: 'ContainerConfig'" compatibility issue
    # Force legacy builder to avoid BuildKit metadata issues with Compose V1
    export DOCKER_BUILDKIT=0
    export COMPOSE_DOCKER_CLI_BUILD=0
    echo -e "${YELLOW}Legacy Mode Enabled: DOCKER_BUILDKIT=0 to prevent ContainerConfig errors.${NC}"

    echo "Attempting to fix known Python dependency issues for legacy docker-compose..."
    # AGGRESSIVE FIX: Downgrade requests/urllib3 to make docker-compose v1 work
    # This addresses 'Not supported URL scheme http+docker' error
    pip install "urllib3<2.0" "requests<2.29.0" --quiet || echo -e "${RED}Failed to auto-fix python dependencies. You might need sudo.${NC}"
fi

BUILD_FLAG=""

# Check arguments
for arg in "$@"; do
    if [ "$arg" == "--build" ]; then
        BUILD_FLAG="--build"
        echo -e "${YELLOW}Build flag detected. Rebuilding image...${NC}"
    fi
done

echo "Executing: $DOCKER_COMPOSE_CMD up -d $BUILD_FLAG"
$DOCKER_COMPOSE_CMD down 
$DOCKER_COMPOSE_CMD up -d $BUILD_FLAG

echo -e "${GREEN}Deployment Successful!${NC}"
echo "Use '$DOCKER_COMPOSE_CMD logs -f' to view logs."
