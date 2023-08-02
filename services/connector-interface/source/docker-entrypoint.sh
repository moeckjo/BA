#!/bin/bash
echo "docker-entrypoint.sh: Starting up"

# Populates the environment variables for the MQTT topics according to the topic convention
# specified in the message format documentation.
echo "docker-entrypoint.sh: Populating environment variables for ${CONNECTOR_INTERFACE_NAME}"

python3 main.py &

# Patches SIGTERM and SIGINT to stop the service. This is required
# to trigger graceful shutdown if docker wants to stop the container.
service_pid=$!
trap "kill -TERM $service_pid" SIGTERM
trap "kill -INT $service_pid" INT

# Run until the container is stopped. Give the service maximal 2 seconds to
# clean up and shut down, afterwards we pull the plug hard.
wait
sleep 2
