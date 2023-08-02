ARG BEM_BASE_IMAGE_TAG
FROM bem-base:$BEM_BASE_IMAGE_TAG
ARG BEM_ROOT_DIR

WORKDIR /bem

# Install python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install GLPK (solver for optimization)
RUN apt-get update && apt-get -y install glpk-utils
RUN glpsol --version

# Copy all necessary data and source code
COPY pv_prediction_models ./pv_prediction_models
COPY source/core ./core
COPY source/tests ./tests

# Install
COPY setup.py .
RUN pip install -e .

# Run tests
# Temporarily copy .env with all variables needed to run the tests
#COPY .env .
#RUN pytest -v ./tests
## The actual .env is mounted when running the container, hence remove this one
#RUN rm .env


WORKDIR core
RUN chown -R bem:bem .

