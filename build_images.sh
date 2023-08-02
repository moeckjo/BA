#!/bin/bash
# This script handles all specialties regarding building all images for an ARMv8 architecture by setting
# some variables prior to built:
# -> For the trading image, "arm64v8-" must be included in the tag to get the image from Blockinfinity for ARM
# -> For the core image, "-aarch64" must be appenden to the Dockerfile file name (Dockerfile-aarch64) to build for ARM
# Setting of the variables using the following script only works if build platform equals target platform

echo -n "Build base images first? (recommended) [y/n]: "
read -r build_base selected_services

touch built_images

if [ "$build_base" = "y" ]; then
  # Build all base images first
    source build_base_images.sh
    # Check if build process has been aborted due to incorrect or missing definition of image tags
    if [ "$continue_build" = "n" ]; then return; fi
    echo "Built base images: $(cat built_images)"

else
  if [ "$selected_services" = "" ]; then
    echo "No images need to be built."
    exit 0
  else
    source set_build_variables.sh

  fi
fi

# Pull required base images from our registry if this script is run in a gitlab CI job
if [ ! -z "$CI" ]; then
  pull_images=()
  # Add base images if needed by specific services
  if [[ "$selected_services" =~ "bem-" ]]; then pull_images+=($BEM_BASE_IMAGE); fi
  if [[ "$selected_services" =~ "bem-core" ]]; then pull_images+=($TENSORFLOW_IMAGE); fi
  if ([[ "$selected_services" =~ "connector" ]] && [[ "$selected_services" != "bem-connector-interface" ]]) || [ "$selected_services" = "all" ]; then pull_images+=("bemcom/python-connector-template:0.1.3" "bemcom/python-connector-template:0.4.0"); fi

  echo "Base images to pull: ${pull_images[*]}"
  for image in "${pull_images[@]}"; do
      echo "Pull $REGISTRY_HOST:$REGISTRY_PORT/$image"
      docker pull $REGISTRY_HOST:$REGISTRY_PORT/$image
      docker tag $REGISTRY_HOST:$REGISTRY_PORT/$image $image
  done
fi

if [ "$selected_services" = "all" ] ; then
  # Get all service names except third party services like InfluxDB
  selected_services="$(docker-compose --log-level ERROR config --services | grep -E "bem-|connector|test")"
  echo "selected_services set from 'all' to: $selected_services"
fi

# Now build the images for all or selected services
echo "Build these bem service images: $selected_services"
docker-compose build $selected_services

for service in $selected_services
do
  # Make service name uppercase and change dash to underscore to get the variable key of the image tag
  image_tag_variable=$(echo $service | tr "[:lower:]" "[:upper:]" | tr "-" "_")_IMAGE_TAG
  # Get value of image tag variable
  image_tag=${!image_tag_variable}

  if [ "$image_tag" = "" ] && [[ "$service" =~ "controller" ]]; then
    echo "Getting shared image for $service ..."
    export $(grep "BEM_CONTROLLER_NAME" .env | xargs)
    service=$BEM_CONTROLLER_NAME
    image_tag_variable=$(echo $service | tr "[:lower:]" "[:upper:]" | tr "-" "_")_IMAGE_TAG
    image_tag=${!image_tag_variable}
  fi

  #  Complete image name
  image="$service:$image_tag"

  if ! [[ "$(cat built_images)" =~ "$image" ]]; then
    echo "Add $image to list of built images."
    echo $image >> built_images
  else
    echo "$image is already on the list."
  fi
done
