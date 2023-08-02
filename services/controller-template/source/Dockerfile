FROM python:3.8

# Add the entrypoint script to run as default.
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod a+x /usr/local/bin/docker-entrypoint.sh

# Installs dependencies.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install the pyconnector package.
COPY controller /source/controller/


ENTRYPOINT ["docker-entrypoint.sh"]
