#\!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Docker Compose deployment for LLM Ops Pipeline${NC}"

# Check if docker-compose is installed
if \! command -v docker-compose &> /dev/null && \! command -v docker &> /dev/null; then
    echo -e "${RED}Neither docker-compose nor docker compose command found. Please install Docker Compose first.${NC}"
    exit 1
fi

# Use Docker Compose to start services
echo -e "${GREEN}Starting services with Docker Compose...${NC}"
if command -v docker-compose &> /dev/null; then
    docker-compose up -d
else
    docker compose up -d
fi

# Check if services are running
echo -e "${GREEN}Checking service status...${NC}"
if command -v docker-compose &> /dev/null; then
    docker-compose ps
else
    docker compose ps
fi

echo -e "${GREEN}Setup complete\!${NC}"
echo -e "Services are available at:"
echo -e "LLM API: ${YELLOW}http://localhost:8000${NC}"
echo -e "MLflow: ${YELLOW}http://localhost:5000${NC}"
echo -e "Prometheus: ${YELLOW}http://localhost:9090${NC}"
echo -e "Grafana: ${YELLOW}http://localhost:3000${NC} (default credentials: admin/admin)"
