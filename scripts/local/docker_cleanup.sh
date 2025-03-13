#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Cleaning up LLM Ops Pipeline Docker Compose deployment${NC}"

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Neither docker-compose nor docker compose command found. Please install Docker Compose first.${NC}"
    exit 1
fi

# Use Docker Compose to stop services
echo -e "${GREEN}Stopping services with Docker Compose...${NC}"
if command -v docker-compose &> /dev/null; then
    docker-compose down
else
    docker compose down
fi

echo -e "${GREEN}Cleanup complete!${NC}"
