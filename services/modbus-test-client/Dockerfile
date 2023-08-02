ARG BEM_BASE_IMAGE_TAG
FROM bem-base:$BEM_BASE_IMAGE_TAG
ENV PYTHONUNBUFFERED 1
ARG BEM_ROOT_DIR

WORKDIR /bem

# Install all third-party packages and copy code
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY source ./modbus_test_client
WORKDIR modbus_test_client





