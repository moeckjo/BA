docker build -t bem-base:$BEM_BASE_IMAGE_TAG -f Dockerfile --build-arg HOST_ARCH_SUFFIX=$HOST_ARCH_SUFFIX --build-arg BEM_BASE_DEPS_WHEELS=wheels --build-arg BEM_ROOT_DIR=$BEM_ROOT_DIR --build-arg DB_CLIENT_IMAGE_TAG=$DB_CLIENT_IMAGE_TAG .

