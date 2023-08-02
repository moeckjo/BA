ARG DB_CLIENT_IMAGE_TAG
FROM bem-db-client:$DB_CLIENT_IMAGE_TAG AS db-image

FROM python:3.8
ENV PYTHONUNBUFFERED 1
#ARG /bem
ARG HOST_ARCH_SUFFIX
ARG BEM_BASE_DEPS_WHEELS

WORKDIR /bem

RUN groupadd -g 999 bem && useradd -r -u 999 -g bem bem
RUN chown -R bem:bem .

# Install all third-party packages
# Load requirements file depending on host architecture (requirements-aarch64.txt for AARCH64/ARM64v8)
COPY requirements${HOST_ARCH_SUFFIX}.txt requirements.txt
RUN mkdir ${BEM_BASE_DEPS_WHEELS}
# Download first to persist wheels, then install packages
RUN pip download -r requirements.txt -d ./${BEM_BASE_DEPS_WHEELS}
RUN pip install -r requirements.txt -f ./${BEM_BASE_DEPS_WHEELS}
# Add all wheels from pip cache (these are the ones without a downloadable wheel!)
RUN cd /root/.cache/pip && find -iname '*.whl' -exec cp {} /bem/${BEM_BASE_DEPS_WHEELS} \;

# Add DB client code
COPY --from=db-image /bem/db /bem/db
COPY --from=db-image /bem/setup.py /bem/setup.py
RUN pip install -e .
RUN mv setup.py setup_db.py

USER bem
RUN export UID
USER root
