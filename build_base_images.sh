# Execute with "source build_base_images.sh" if you want to keep the build variables, "bash build_base_images" otherwise.

# Builds image with DB client code, then the bem-base image in a multi-stage build, including th DB code
# Additionally, for ARM, the tensorflow image is built.

# This script handles all specialties regarding building all images for an ARMv8 architecture by setting
# some variables prior to built (see comments in script for details).
# Setting of the variables using the following script only works if build platform equals target platform
source set_build_variables.sh

echo -n "All image tags set correctly? Type 'y' to continue build or 'n' to abort.: "
read -r continue_build
if [ "$continue_build" = "n" ]; then
  export continue_build=$continue_build
  echo "Aborting build process. Please define tags in set_build_variables.sh"
  return
fi

echo "Build bem-db-client"
#cd services/db-client
cd bem_base_images/db-client
bash build_docker_image.sh

echo "Build bem-base"
#cd ../../bem_base_images/bem-base
cd ../bem-base
bash build_docker_image.sh

if [ "$(uname -m)" = "aarch64" ]; then
  cd ../tensorflow-arm64v8
  bash build_docker_image.sh
fi

cd ../../
echo $DB_CLIENT_IMAGE >> built_images
echo $BEM_BASE_IMAGE >> built_images
if [ "$(uname -m)" = "aarch64" ]; then echo $TENSORFLOW_IMAGE >> built_images; fi

echo "Build of base images finished."
