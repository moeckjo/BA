# Empty string for amd64 or x86-64 (default), "arm64v8-" for ARM64 -> see comment in docker-compose entry
ARG HOST_ARCH
ARG BEM_BASE_IMAGE_TAG
FROM bem-base:$BEM_BASE_IMAGE_TAG as base

ARG BEM_ROOT_DIR
ARG BEM_BASE_DEPS_WHEELS_ARCHIVE
ARG BEM_BASE_DEPS_WHEELS

WORKDIR /bem
RUN python setup_db.py bdist_wheel -d .
RUN mv db-1.0-py3-none-any.whl wheels/db-1.0-py3-none-any.whl
# Make archive with wheels of bem-base dependencies + db package
RUN tar vczf ${BEM_BASE_DEPS_WHEELS_ARCHIVE} ${BEM_BASE_DEPS_WHEELS}

FROM blockinfinityhub/flexqgridpythonsdk:${HOST_ARCH}3.8-buster-latest-sdk
ARG BEM_ROOT_DIR
ARG BEM_BASE_DEPS_WHEELS_ARCHIVE
ARG BEM_BASE_DEPS_WHEELS

WORKDIR /bem
RUN groupadd -g 999 bem && useradd -r -u 999 -g bem bem

# Copy wheel archive and requirements file from bem-base
COPY --from=base /bem/${BEM_BASE_DEPS_WHEELS_ARCHIVE} ./${BEM_BASE_DEPS_WHEELS_ARCHIVE}
#COPY --from=base /bem/requirements.txt ./requirements_bem.txt
## Add db package to requirements
#RUN echo db >> requirements_bem.txt
RUN tar vzxf ${BEM_BASE_DEPS_WHEELS_ARCHIVE}

# Install python requirements
COPY requirements.txt .
RUN echo db >> requirements.txt
RUN pip install -r requirements.txt -f ${BEM_BASE_DEPS_WHEELS} --no-index

## Decompress wheel archive and install the packages
#RUN pip install -r requirements_bem.txt -f ${BEM_BASE_DEPS_WHEELS} --no-index

# Remove all wheels to shrink image size
RUN rm -r /bem/${BEM_BASE_DEPS_WHEELS}
RUN rm ${BEM_BASE_DEPS_WHEELS_ARCHIVE}

# Copy all necessary data and source code
COPY source/trading ./trading

# Install
COPY setup.py .
RUN pip install -e .

WORKDIR trading
COPY start.sh .
RUN chown -R bem:bem .

USER bem
RUN export UID
USER root
