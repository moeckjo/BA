To install the DB client code from the bem-db-client image inside another container, do the following:

### 1. Export build variables
Export these variables in your shell.
Note: Adjust the tag according to the actual tag of the bem-db-client image tag
```
export BEM_ROOT_DIR="/bem"
export TAG=0.1
``` 

### 2. Dockerfile
Add this to your Dockerfile
```
ARG TAG=$TAG
FROM bem-db-client:$TAG AS db-image

FROM final-image-name:final-image-tag
ARG BEM_ROOT_DIR=$BEM_ROOT_DIR

# Install DB client code
COPY --from=db-image /bem/db /bem/db
COPY --from=db-image /bem/setup.py /bem/setup.py
RUN pip install -e .
# (Optional) Rename setup.py to avoid confusion with other setup.py files in your service
RUN mv setup.py setup_db.py 

# Your stuff

```
