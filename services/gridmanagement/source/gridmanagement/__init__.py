import logging
import os
import sys

# Note (from docs: https://docs.python.org/3/library/logging.html#logger-objects):
# Multiple calls to getLogger() with the same name will always return a reference to the same Logger object.
logger = logging.getLogger('bem-gridmanagement')
loglevel = logging.DEBUG if os.getenv("DEBUG").lower() == "true" else logging.INFO
logger.setLevel(loglevel)
logformat = '[%(asctime)s - %(name)s - task:%(task_name)s - %(funcName)s - %(levelname)s]:  %(message)s'
logging.getLogger("pika").setLevel(logging.WARNING)