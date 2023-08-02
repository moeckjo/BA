ARG BEM_BASE_IMAGE_TAG
FROM bem-base:$BEM_BASE_IMAGE_TAG
ARG BEM_ROOT_DIR

WORKDIR /bem

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY source/orchestration ./orchestration
WORKDIR orchestration




