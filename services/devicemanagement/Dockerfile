ARG BEM_BASE_IMAGE_TAG
FROM bem-base:$BEM_BASE_IMAGE_TAG
ARG BEM_ROOT_DIR

WORKDIR /bem

# Install python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY source/devicemanagement ./devicemanagement
COPY source/tests ./tests

# Install
COPY setup.py .
RUN pip install -e .

# Run tests
# Temporarily copy .env with all variables and config needed to run the tests
COPY .env .
COPY config ./config
RUN pytest -v ./tests
# The actual .env and config is mounted when running the container, hence remove this one
RUN rm .env
RUN rm -r ./config

WORKDIR devicemanagement
RUN chown -R bem:bem .
