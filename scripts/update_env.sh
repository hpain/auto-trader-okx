#!/bin/bash
# scripts/update_env.sh

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== OKX API Key Manager ===${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}.env file not found. copying from .env.example...${NC}"
    cp .env.example .env 2>/dev/null || touch .env
fi

echo "Current .env status:"
if grep -q "OKX_API_KEY=" .env; then
    echo -e "  OKX_API_KEY: ${GREEN}Found${NC}"
else
    echo -e "  OKX_API_KEY: ${RED}Missing${NC}"
fi

echo ""
read -p "Do you want to update your API Keys now? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting."
    exit 0
fi

# Function to update or append key
update_key() {
    local key=$1
    local value=$2
    local file=$3
    
    # Escape special characters for sed (specifically / &)
    # But simple way is to delete old line and append new one
    
    # Check if key exists
    if grep -q "^$key=" "$file"; then
        # Delete existing line (using a temp file strategy for compatibility)
        grep -v "^$key=" "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    fi
    
    # Append new key
    echo "$key=$value" >> "$file"
    echo -e "Updated $key"
}

read -p "Enter OKX API Key: " api_key
read -p "Enter OKX Secret Key: " secret_key
read -p "Enter OKX Passphrase: " passphrase

echo -e "\n${YELLOW}Updating .env file...${NC}"

update_key "OKX_API_KEY" "$api_key" ".env"
update_key "OKX_SECRET_KEY" "$secret_key" ".env"
update_key "OKX_PASSPHRASE" "$passphrase" ".env"

# Also update LIVE keys for Arb Bot just in case
update_key "LIVE_OKX_API_KEY" "$api_key" ".env"
update_key "LIVE_OKX_SECRET_KEY" "$secret_key" ".env"
update_key "LIVE_OKX_PASSPHRASE" "$passphrase" ".env"

echo -e "${GREEN}Success! .env updated.${NC}"
echo "You can now restart your containers with: ./deploy/start.sh"
