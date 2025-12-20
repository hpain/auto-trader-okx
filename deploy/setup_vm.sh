#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Environment Setup...${NC}"

# Update system
echo "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install Git
if ! command -v git &> /dev/null; then
    echo "Installing Git..."
    sudo apt-get install -y git
else
    echo "Git is already installed."
fi

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Add current user to docker group to avoid using sudo for docker commands
    echo "Adding current user to docker group..."
    sudo usermod -aG docker $USER
    echo "PLEASE NOTE: You may need to log out and log back in for docker group changes to take effect."
else
    echo "Docker is already installed."
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    # Check if docker-compose-plugin is available (newer method)
    if apt-cache search docker-compose-plugin | grep -q docker-compose-plugin; then
         sudo apt-get install -y docker-compose-plugin
    else
         sudo apt-get install -y docker-compose
    fi
else
    echo "Docker Compose is already installed."
fi

echo -e "${GREEN}Setup Complete!${NC}"
echo "You may need to logout and login again if this is the first time you installed Docker."
