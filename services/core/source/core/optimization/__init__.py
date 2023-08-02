import datetime
import os
import random
import bson
import pandas
import numpy as np
import pytz
import typing
import json

import db.db_helper as db

import logging

logger = logging.getLogger('bem-core.optimization')
loglevel = logging.DEBUG if os.getenv("DEBUG_OPTIMIZATION").lower() == "true" else logging.INFO
logger.setLevel(loglevel)
logging.getLogger("pika").setLevel(logging.WARNING)
