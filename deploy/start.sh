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
echo -e "${GREEN}Building and Starting Docker Containers...${NC}"
docker-compose down # Stop existing if any
docker-compose up -d --build

echo -e "${GREEN}Deployment Successful!${NC}"
echo "Use 'docker-compose logs -f' to view logs."
