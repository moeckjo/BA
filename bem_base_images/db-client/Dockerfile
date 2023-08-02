FROM python:3.8
ENV PYTHONUNBUFFERED 1
ARG BEM_ROOT_DIR=$BEM_ROOT_DIR

WORKDIR /bem

COPY source/db ./db
COPY setup.py .
RUN pip install -e .




