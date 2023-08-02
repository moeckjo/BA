FROM bemcom/python-connector-template:0.1.3

# Installs dependencies.
COPY requirements.txt /source/requirements.txt
RUN pip install -r /source/requirements.txt

# Export your connector script. Ensure the main program of the
# connector resides under ./source/connector/main.py
COPY connector /source/connector/

# Copy configuration files into the image
# COPY config ./config/
RUN mkdir /source/connector/config

WORKDIR /source/connector
