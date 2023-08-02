# Note: script must be executed with 'source set_...' or '. set_...', not 'bash set_...' to export variables to current shell
# DON'T CHANGE ANYTHING BELOW
echo "Setting build variables."

# -> For the core image, "-aarch64" must be appended to the Dockerfile file name (Dockerfile-aarch64) and
# requirements file to build for ARM
# -> For the trading image, "arm64v8-" must be included in the tag to get the image from Blockinfinity for ARM

# This statement only works if build platform equals target platform
if [ "$(uname -m)" = "aarch64" ]; then export HOST_ARCH_SUFFIX="-aarch64"; else export HOST_ARCH_SUFFIX=""; fi
echo "Variable HOST_ARCH_SUFFIX set to '$HOST_ARCH_SUFFIX' (empty string if not building for ARM)"
if [ "$(uname -m)" = "aarch64" ]; then export HOST_ARCH_TRADING_SERVICE="arm64v8-"; else export HOST_ARCH_TRADING_SERVICE=""; fi
echo "Variable HOST_ARCH_TRADING_SERVICE set to '$HOST_ARCH_TRADING_SERVICE' (empty string if not building for ARM)"

export BEM_ROOT_DIR="/bem"
#echo "Root directory of all services set to $BEM_ROOT_DIR"

# Get image version tag from associated git tag and append branch name if applicable
DB_CLIENT_IMAGE_TAG=$(git tag -l 'bem-db-client_*' | sort -V | tail -n1 | cut -d_ -f2)
BEM_BASE_IMAGE_TAG=$(git tag -l 'bem-base_*' | sort -V | tail -n1 | cut -d_ -f2)
TENSORFLOW_IMAGE_TAG=$(git tag -l 'ubuntu-tensorflow/arm64v8_*' | sort -V | tail -n1 | cut -d_ -f2)

echo "Current setting of base image tags:"

export DB_CLIENT_IMAGE_TAG=$DB_CLIENT_IMAGE_TAG
export DB_CLIENT_IMAGE="bem-db-client:$DB_CLIENT_IMAGE_TAG"
echo $DB_CLIENT_IMAGE

export BEM_BASE_IMAGE_TAG=$BEM_BASE_IMAGE_TAG
export BEM_BASE_IMAGE="bem-base:$BEM_BASE_IMAGE_TAG"
echo $BEM_BASE_IMAGE

if [ "$(uname -m)" = "aarch64" ]; then
  export TENSORFLOW_IMAGE_TAG=$TENSORFLOW_IMAGE_TAG
  export TENSORFLOW_IMAGE="ubuntu-tensorflow/arm64v8:$TENSORFLOW_IMAGE_TAG"
  echo $TENSORFLOW_IMAGE
fi

if [ ! "$CI_COMMIT_BRANCH" = "" ] && [ ! "$CI_COMMIT_BRANCH" = "master" ] ; then BRANCH_TAG="-$CI_COMMIT_BRANCH"; fi

# Get image version tag from associated git tag and append branch name if applicable
export BEM_CORE_IMAGE_TAG=$(git tag -l 'bem-core_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG
export BEM_GRIDMANAGEMENT_IMAGE_TAG=$(git tag -l 'bem-gridmanagement_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG
export BEM_DEVICEMANAGEMENT_IMAGE_TAG=$(git tag -l 'bem-devicemanagement_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG
export BEM_TRADING_IMAGE_TAG=$(git tag -l 'bem-trading_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG
export BEM_ORCHESTRATION_IMAGE_TAG=$(git tag -l 'bem-orchestration_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG
export BEM_MODBUS_SERVER_IMAGE_TAG=$(git tag -l 'bem-modbus-server_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG
export BEM_MODBUS_TEST_CLIENT_IMAGE_TAG=$(git tag -l 'bem-modbus-test-client_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG

export BEM_CONNECTOR_INTERFACE_IMAGE_TAG=$(git tag -l 'bem-connector-interface_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG

export BEM_CONTROLLER_IMAGE_TAG=$(git tag -l 'bem-controller_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG

export MODBUS_TCP_CONNECTOR_KEBA_P30_IMAGE_TAG=$(git tag -l 'modbus-tcp-connector-keba-p30_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG
export RSCP_CONNECTOR_E3DC_IMAGE_TAG=$(git tag -l 'rscp-connector-e3dc_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG

export MQTT_TEST_LISTENER_IMAGE_TAG=$(git tag -l 'mqtt-test-listener_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG
export MQTT_TEST_PUBLISHER_IMAGE_TAG=$(git tag -l 'mqtt-test-publisher_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG

export RABBITMQ_TEST_PUBLISHER_IMAGE_TAG=$(git tag -l 'rabbitmq-test-publisher_*' | sort -V | tail -n1 | cut -d_ -f2)$BRANCH_TAG
