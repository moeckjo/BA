import datetime
import os
import sys

import bson
import pandas
import numpy as np
import pytz
import typing
import json
import logging


import db.db_helper as db

logger = logging.getLogger('bem-core.forecasting')
loglevel = logging.DEBUG if os.getenv("DEBUG_FORECASTING").lower() == "true" else logging.INFO
logger.setLevel(loglevel)
logformat = '[%(asctime)s - %(name)s - task:%(task_name)s - %(funcName)s - %(levelname)s]:  %(message)s'

path = os.getenv('BEM_ROOT_DIR')
