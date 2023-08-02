ARG BEM_BASE_IMAGE_TAG
FROM bem-base:$BEM_BASE_IMAGE_TAG
ARG BEM_ROOT_DIR

WORKDIR /bem

# Installs dependencies.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Export your connector script. Ensure the main program of the
# connector resides under ./source/connector/main.py
COPY interface ./interface/

# Copy configuration files into the image
# COPY config ./config/
RUN mkdir ./interface/config

# Install
COPY setup.py .
RUN pip install -e .

WORKDIR ./interface/

# Add the entrypoint script to run as default.
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod a+x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]