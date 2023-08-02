ARG BEM_BASE_IMAGE_TAG
FROM bem-base:$BEM_BASE_IMAGE_TAG
ARG BEM_ROOT_DIR

WORKDIR /bem

# Install all third-party packages and copy code
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY source/modbus-server ./modbus-server
WORKDIR modbus-server





