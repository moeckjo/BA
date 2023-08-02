import logging
import os
import sys

logger = logging.getLogger('bem-devicemanagement')
loglevel = logging.DEBUG if os.getenv("DEBUG").lower() == "true" else logging.INFO
logger.setLevel(loglevel)
logformat = '[%(asctime)s - %(name)s - task:%(task_name)s - %(funcName)s - %(levelname)s]:  %(message)s'
logging.getLogger("pika").setLevel(logging.WARNING)