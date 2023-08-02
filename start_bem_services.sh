#!/bin/bash
# File may exist if bem-orchestration crashed. Causes problems if not removed before start.
FILE=services/orchestration/celerybeat.pid
if [ -f "$FILE" ]; then
  sudo rm $FILE
fi

# Need to export image tags
source set_build_variables.sh
export UID GID # needed for databases
# Get variable that indicates if the quota market is active, which necessitates the trading service
export $(grep "QUOTA_MARKET_ENABLED" .env | xargs)

if [ $QUOTA_MARKET_ENABLED = 1 ] || [ $(echo "$QUOTA_MARKET_ENABLED" | tr '[:upper:]' '[:lower:]')  == "true" ]; then
  docker-compose up $1 bem-core bem-devicemanagement bem-gridmanagement bem-orchestration bem-trading
else
  docker-compose up $1 bem-core bem-devicemanagement bem-gridmanagement bem-orchestration
fi

