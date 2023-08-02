#!/bin/bash
docker-compose stop bem-core bem-devicemanagement bem-gridmanagement bem-orchestration bem-trading bem-modbus-server
if [ "$1" = "-rm" ]; then
  docker rm bem-core bem-devicemanagement bem-gridmanagement bem-orchestration bem-trading
fi
