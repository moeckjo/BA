FROM python:3.8

# Installs dependencies.
COPY requirements.txt /source/requirements.txt
RUN pip install -r /source/requirements.txt

# Export your connector script. Ensure the main program of the
# connector resides under ./source/connector/main.py
COPY connector /source/connector/

# Add the entrypoint script to run as default.
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod a+x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]